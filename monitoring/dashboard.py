"""Production Observability Dashboard for Autonomous Agent System.

Supports dual modes:
1. Rich Terminal CLI Dashboard: Live refreshing terminal UI for headless or SSH operations.
2. Streamlit Web Dashboard: Interactive browser-based telemetry with charts and tables.

Displays:
- Real-time ingestion rate & total papers cataloged.
- FAISS memory vector growth and persistence state.
- Live ReAct session throughput, average steps-to-resolution, and self-correction retry frequencies.
- System resource gauges (RAM, VRAM, and unwritten buffer queue).
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from rich.align import Align
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from config import get_settings
from memory.vector_store import VectorStore
from monitoring.metrics import update_system_gauges
from storage.mysql_manager import get_mysql_manager


def fetch_telemetry_snapshot() -> Dict[str, Any]:
    """Collect current system telemetry across database, vector store, and system resources."""
    settings = get_settings()
    mysql_mgr = get_mysql_manager(settings=settings)
    vector_store = VectorStore(settings=settings)

    # 1. Vector Memory Stats
    try:
        vector_store.load_local()
        total_vectors = vector_store.index.ntotal if vector_store.index else 0
        embedding_dim = vector_store.dimension
        index_file = settings.memory_local_dir / "index.faiss"
        index_size_kb = round(index_file.stat().st_size / 1024, 1) if index_file.exists() else 0.0
    except Exception:
        total_vectors = 0
        embedding_dim = 384
        index_size_kb = 0.0

    # 2. Relational Database Stats
    total_papers = 0
    papers_by_source: Dict[str, int] = {}
    total_sessions = 0
    success_sessions = 0
    avg_steps = 0.0
    total_retries = 0
    avg_latency = 0.0
    recent_sessions: List[Dict[str, Any]] = []

    conn = mysql_mgr.get_connection()
    if conn:
        try:
            cursor = conn.cursor(dictionary=True)
            # Papers count
            cursor.execute("SELECT COUNT(*) as count FROM research_papers")
            total_papers = cursor.fetchone().get("count", 0)

            # Papers by source
            cursor.execute("SELECT source, COUNT(*) as count FROM research_papers GROUP BY source")
            for row in cursor.fetchall():
                papers_by_source[row.get("source", "unknown")] = row.get("count", 0)

            # Sessions
            cursor.execute(
                "SELECT COUNT(*) as count, "
                "SUM(CASE WHEN status='success' THEN 1 ELSE 0 END) as successes, "
                "AVG(total_steps) as avg_steps, "
                "SUM(retry_count) as total_retries, "
                "AVG(execution_latency) as avg_latency "
                "FROM agent_sessions"
            )
            s_row = cursor.fetchone() or {}
            total_sessions = s_row.get("count") or 0
            success_sessions = s_row.get("successes") or 0
            avg_steps = round(float(s_row.get("avg_steps") or 0.0), 2)
            total_retries = s_row.get("total_retries") or 0
            avg_latency = round(float(s_row.get("avg_latency") or 0.0), 2)

            # Recent sessions
            cursor.execute(
                "SELECT session_id, input_objective, status, total_steps, retry_count, execution_latency "
                "FROM agent_sessions ORDER BY session_timestamp DESC LIMIT 5"
            )
            recent_sessions = cursor.fetchall() or []

            cursor.close()
            conn.close()
        except Exception:
            if conn and conn.is_connected():
                conn.close()

    # 3. System resource gauges
    gauges = update_system_gauges(
        active_vectors=total_vectors,
        buffered_count=mysql_mgr.buffered_count,
    )

    success_rate = round((success_sessions / total_sessions * 100), 1) if total_sessions > 0 else 100.0

    return {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "total_papers": total_papers,
        "papers_by_source": papers_by_source,
        "total_vectors": total_vectors,
        "embedding_dim": embedding_dim,
        "index_size_kb": index_size_kb,
        "total_sessions": total_sessions,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "total_retries": total_retries,
        "avg_latency": avg_latency,
        "recent_sessions": recent_sessions,
        "buffered_count": mysql_mgr.buffered_count,
        "ram_mb": round(gauges["ram_bytes"] / (1024 * 1024), 1),
        "vram_mb": round(gauges["vram_bytes"] / (1024 * 1024), 1),
    }


# ===========================================================================
# 1. Rich Terminal CLI Dashboard
# ===========================================================================

def generate_cli_dashboard(snapshot: Dict[str, Any]) -> Layout:
    """Construct Rich Layout with telemetry panels and status cards."""
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="metrics", size=8),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3),
    )

    # Header
    header_text = Text()
    header_text.append(" AGI AUTONOMOUS AGENT ", style="bold black on green")
    header_text.append(f"  |  Operational Telemetry & Ingestion Dashboard  |  Updated: {snapshot['timestamp']}", style="dim")
    layout["header"].update(Panel(header_text, style="green"))

    # Top Metric Cards (3x columns)
    layout["metrics"].split_row(
        Layout(name="papers_card"),
        Layout(name="vector_card"),
        Layout(name="agent_card"),
        Layout(name="system_card"),
    )

    # Papers Card
    papers_table = Table.grid(padding=(0, 1))
    papers_table.add_row("Total Papers:", f"[bold cyan]{snapshot['total_papers']}[/]")
    for src, cnt in list(snapshot["papers_by_source"].items())[:3]:
        papers_table.add_row(f"  - {src}:", f"[dim]{cnt}[/]")
    layout["papers_card"].update(Panel(papers_table, title="[bold]Ingestion Stream[/]", border_style="cyan"))

    # Vector Memory Card
    vector_table = Table.grid(padding=(0, 1))
    vector_table.add_row("Indexed Vectors:", f"[bold magenta]{snapshot['total_vectors']}[/]")
    vector_table.add_row("Vector Dimension:", f"[dim]{snapshot['embedding_dim']}[/]")
    vector_table.add_row("Disk Index Size:", f"[dim]{snapshot['index_size_kb']} KB[/]")
    layout["vector_card"].update(Panel(vector_table, title="[bold]FAISS Vector Memory[/]", border_style="magenta"))

    # Agent ReAct Card
    agent_table = Table.grid(padding=(0, 1))
    agent_table.add_row("Completed Sessions:", f"[bold green]{snapshot['total_sessions']}[/]")
    agent_table.add_row("Success Rate:", f"[bold green]{snapshot['success_rate']}%[/]")
    agent_table.add_row("Avg Steps / Session:", f"[dim]{snapshot['avg_steps']}[/]")
    agent_table.add_row("Self-Correction Retries:", f"[dim yellow]{snapshot['total_retries']}[/]")
    layout["agent_card"].update(Panel(agent_table, title="[bold]ReAct Agent Cycles[/]", border_style="green"))

    # System Health Card
    sys_table = Table.grid(padding=(0, 1))
    sys_table.add_row("System RAM Usage:", f"[bold yellow]{snapshot['ram_mb']} MB[/]")
    sys_table.add_row("GPU VRAM Usage:", f"[dim]{snapshot['vram_mb']} MB[/]")
    sys_table.add_row("MySQL In-Memory Buffer:", f"[bold red]{snapshot['buffered_count']}[/]" if snapshot['buffered_count'] > 0 else "[green]0 (healthy)[/]")
    layout["system_card"].update(Panel(sys_table, title="[bold]Resource Telemetry[/]", border_style="yellow"))

    # Main Area: Recent Sessions Table
    sessions_table = Table(title="Recent Agent Reasoning Sessions", expand=True)
    sessions_table.add_column("Session ID", style="cyan", no_wrap=True)
    sessions_table.add_column("Input Objective", style="white")
    sessions_table.add_column("Status", style="bold")
    sessions_table.add_column("Steps", justify="right")
    sessions_table.add_column("Retries", justify="right")
    sessions_table.add_column("Latency (s)", justify="right")

    for s in snapshot["recent_sessions"]:
        status_style = "green" if s.get("status") == "success" else "red"
        sessions_table.add_row(
            str(s.get("session_id", ""))[:12] + "...",
            str(s.get("input_objective", ""))[:65] + "...",
            f"[{status_style}]{s.get('status', 'unknown')}[/]",
            str(s.get("total_steps", 0)),
            str(s.get("retry_count", 0)),
            f"{s.get('execution_latency', 0.0):.2f}",
        )

    if not snapshot["recent_sessions"]:
        sessions_table.add_row("[dim]No sessions recorded yet[/]", "-", "-", "-", "-", "-")

    layout["main"].update(Panel(sessions_table, border_style="blue"))

    # Footer
    footer_text = Text("Press Ctrl+C to exit  |  Prometheus Scrape: http://localhost:8000/metrics  |  Daemon: http://localhost:9090", style="dim italic")
    layout["footer"].update(Panel(Align.center(footer_text), style="dim"))

    return layout


def run_cli_dashboard(refresh_rate_seconds: float = 2.0, once: bool = False) -> None:
    """Run interactive terminal dashboard using Rich."""
    console = Console()
    if once:
        snapshot = fetch_telemetry_snapshot()
        console.print(generate_cli_dashboard(snapshot))
        return

    console.clear()
    with Live(console=console, screen=True, refresh_per_second=2) as live:
        try:
            while True:
                snapshot = fetch_telemetry_snapshot()
                live.update(generate_cli_dashboard(snapshot))
                time.sleep(refresh_rate_seconds)
        except KeyboardInterrupt:
            pass


# ===========================================================================
# 2. Streamlit Web Dashboard Mode
# ===========================================================================

def render_streamlit_dashboard() -> None:
    """Render interactive web dashboard if executed with `streamlit run`."""
    try:
        import streamlit as st  # type: ignore[import-not-found,reportMissingImports]
    except ImportError:
        print("Streamlit is not installed. To run web dashboard: uv pip install streamlit")
        return

    st.set_page_config(page_title="AGI Agent Observability", page_icon="🤖", layout="wide")
    st.title("🤖 AGI Autonomous Agent: Observability & Telemetry")

    snapshot = fetch_telemetry_snapshot()

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cataloged Papers", snapshot["total_papers"])
    c2.metric("FAISS Memory Vectors", snapshot["total_vectors"], f"{snapshot['index_size_kb']} KB")
    c3.metric("Agent Success Rate", f"{snapshot['success_rate']}%", f"{snapshot['total_sessions']} sessions")
    c4.metric("Process RAM", f"{snapshot['ram_mb']} MB", f"VRAM: {snapshot['vram_mb']} MB")

    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📚 Literature Ingestion Breakdown")
        if snapshot["papers_by_source"]:
            st.bar_chart(snapshot["papers_by_source"])
        else:
            st.info("No papers cataloged in relational audit database yet.")

    with col_right:
        st.subheader("⚡ ReAct Reasoning Performance")
        st.write(f"**Average Steps to Resolution:** {snapshot['avg_steps']}")
        st.write(f"**Average Execution Latency:** {snapshot['avg_latency']} seconds")
        st.write(f"**Self-Correction Retries:** {snapshot['total_retries']}")
        st.write(f"**MySQL In-Memory Buffer Size:** {snapshot['buffered_count']} items")

    st.divider()
    st.subheader("🔍 Recent Agent Sessions")
    if snapshot["recent_sessions"]:
        st.dataframe(snapshot["recent_sessions"], use_container_width=True)
    else:
        st.info("No agent sessions recorded in MySQL database yet.")


# ===========================================================================
# CLI Dispatcher
# ===========================================================================

if __name__ == "__main__":
    # Check if running under Streamlit
    if "streamlit" in sys.modules or os.environ.get("STREAMLIT_RUN") == "1":
        render_streamlit_dashboard()
    else:
        parser = argparse.ArgumentParser(description="AGI System Observability Dashboard")
        parser.add_argument("--once", action="store_true", help="Print a single telemetry snapshot and exit")
        parser.add_argument("--refresh", type=float, default=2.0, help="Refresh interval in seconds (default 2.0)")
        parser.add_argument("--web", action="store_true", help="Launch Streamlit web dashboard")
        args = parser.parse_args()

        if args.web:
            render_streamlit_dashboard()
        else:
            run_cli_dashboard(refresh_rate_seconds=args.refresh, once=args.once)
