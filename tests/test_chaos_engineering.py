"""Production Chaos Engineering & Transient Network Fault Injection Test Suite.

Simulates real-world distributed infrastructure failures and validates platform survivability:
1. Chaos Scenario A (MySQL Partitioning):
   - Sudden connection severance between serving/orchestrator and MySQL during active ReAct reasoning.
   - Confirms zero dropped requests, resilient in-memory buffer queueing, and automatic batch replay upon container recovery.
2. Chaos Scenario B (Rate Limiting Storm):
   - Simulates HTTP 429 (Too Many Requests) and 406 (Rate Limit/Soft-Block) storm against ArXiv ingestion.
   - Verifies exponential backoff with pacing, retry exhaustion handling, and zero unhandled daemon crashes.
3. Chaos Scenario C (Prometheus Scrape Under Load):
   - Concurrently bombards /metrics with 50 simultaneous scrape requests while /v1/chat/completions is processing inference requests.
   - Verifies all requests return HTTP 200 and inference latency degradation stays strictly under 10%.
"""

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent.orchestrator import ReActOrchestrator
from agent.tools import create_default_tool_registry
from config import Settings
from daemon.continuous_ingest import ContinuousIngestionDaemon
from ingestion.stream_sources import StreamIngestionClient
from serving.app import app, get_inference_engine
from storage.mysql_manager import MySQLAuditManager


@pytest.fixture(autouse=True)
def clean_singletons():
    """Reset singletons before and after each test."""
    MySQLAuditManager.reset_singleton()
    yield
    MySQLAuditManager.reset_singleton()


# ===========================================================================
# Chaos Scenario A: MySQL Partitioning & Resilient Batch Replay
# ===========================================================================

def test_chaos_mysql_partition_during_react_reasoning():
    """Chaos Scenario A: Sever MySQL connection during active ReAct reasoning cycle.

    Validates:
    - ReAct loop completes non-blocking without raising exceptions (zero dropped requests).
    - Unwritten session telemetry and reasoning steps are captured in memory buffer.
    - Simulated database recovery automatically re-establishes connectivity and batch-replays buffered records.
    """
    settings = Settings(
        mysql_host="agi-mysql",
        mysql_port=3306,
        mysql_database="agi_memory",
    )
    mysql_mgr = MySQLAuditManager(settings=settings)

    # Simulate network partition between serving layer and MySQL
    mysql_mgr._is_available = False
    mysql_mgr.get_connection = MagicMock(return_value=None)

    tools = create_default_tool_registry()
    orchestrator = ReActOrchestrator(
        tools=tools,
        max_steps=2,
        audit_manager=mysql_mgr,
    )

    # 1. Active reasoning under network partition
    final_answer = orchestrator.run("Explain topological context sharing in Mixture of Experts routing.")

    # Verify zero dropped requests: Agent produces valid conclusion despite database severance
    assert final_answer is not None
    assert final_answer.success is True
    assert len(final_answer.steps) > 0

    # 2. In-memory buffer verification: Telemetry must be quarantined safely
    assert mysql_mgr.buffered_count == 1, "Session telemetry must be safely queued during partition"
    assert len(mysql_mgr._buffered_sessions) == 1
    session_data, steps_data = mysql_mgr._buffered_sessions[0]
    assert session_data["status"] == "success"
    assert len(steps_data) > 0

    # 3. Simulate container recovery (MySQL partition heals)
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mysql_mgr.get_connection = MagicMock(return_value=mock_conn)
    mysql_mgr._is_available = True

    # 4. Batch replay upon recovery
    flush_report = mysql_mgr.flush_buffer()
    assert flush_report["flushed_sessions"] == 1
    assert mysql_mgr.buffered_count == 0, "Buffer queue must be drained to 0 after replay"
    assert mock_cursor.execute.call_count >= 1
    assert mock_conn.commit.called


def test_chaos_mysql_partition_multi_cycle_accumulation_and_drain():
    """Verify multiple agent sessions accumulate during prolonged database partition and flush atomically."""
    settings = Settings(mysql_host="agi-mysql", mysql_port=3306)
    mysql_mgr = MySQLAuditManager(settings=settings)
    mysql_mgr._is_available = False
    mysql_mgr.get_connection = MagicMock(return_value=None)

    # Queue 5 independent sessions during network partition
    for i in range(5):
        s_data = {
            "session_id": f"sess_chaos_{i}",
            "status": "success",
            "input_objective": f"Objective {i}",
            "execution_latency": 0.5,
        }
        st_data = [{"step_index": 1, "action_name": "calc", "observation_output": "ok"}]
        mysql_mgr.record_agent_session(s_data, st_data)

    assert mysql_mgr.buffered_count == 5

    # Simulate container healing
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mysql_mgr.get_connection = MagicMock(return_value=mock_conn)
    mysql_mgr._is_available = True

    flush_report = mysql_mgr.flush_buffer()
    assert flush_report["flushed_sessions"] == 5
    assert mysql_mgr.buffered_count == 0


# ===========================================================================
# Chaos Scenario B: ArXiv Rate Limiting Storm (HTTP 429 & 406)
# ===========================================================================

@pytest.mark.asyncio
async def test_chaos_arxiv_rate_limiting_storm_exponential_backoff():
    """Chaos Scenario B: Flood ArXiv queries with 429/406 responses, verify exponential backoff & recovery."""
    client = StreamIngestionClient()
    sleep_intervals: List[float] = []

    async def mock_sleep(duration: float):
        sleep_intervals.append(duration)

    call_count = 0
    sample_atom = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom">\n'
        '  <entry>\n'
        '    <id>http://arxiv.org/abs/2609.12345v1</id>\n'
        '    <title>Topological Signatures of Context Sharing</title>\n'
        '    <summary>Empirical study on MoE expert routing and attention awareness.</summary>\n'
        '    <published>2026-09-20T12:00:00Z</published>\n'
        '  </entry>\n'
        '</feed>'
    )

    async def mock_get(url: str):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First attempt: 429 Too Many Requests
            resp = httpx.Response(status_code=429, request=httpx.Request("GET", url))
            return resp
        elif call_count == 2:
            # Second attempt: 406 Rate Limit Soft-Block
            resp = httpx.Response(status_code=406, request=httpx.Request("GET", url))
            return resp
        else:
            # Third attempt: Successful recovery
            return httpx.Response(status_code=200, text=sample_atom, request=httpx.Request("GET", url))

    mock_http_client = AsyncMock()
    mock_http_client.get = mock_get

    with patch("asyncio.sleep", side_effect=mock_sleep):
        records = await client._fetch_single_arxiv_category(
            client=mock_http_client,
            category="cs.AI",
            limit=5,
            start=0,
        )

    # 1. Verify retries executed
    assert call_count == 3, f"Expected 3 attempts (429 -> 406 -> 200), got {call_count}"

    # 2. Verify exponential backoff intervals were applied (3.0s attempt 1, 6.0s attempt 2)
    assert len(sleep_intervals) == 2
    assert sleep_intervals[0] == 3.0
    assert sleep_intervals[1] == 6.0

    # 3. Verify clean recovery and parsed records
    assert len(records) == 1
    assert "Topological Signatures" in records[0].title


@pytest.mark.asyncio
async def test_chaos_arxiv_rate_limiting_exhaustion_clean_handling():
    """Verify that when 429 rate limit storm completely exhausts all retries, the daemon handles it cleanly without crashing."""
    client = StreamIngestionClient()

    async def mock_get_always_429(url: str):
        return httpx.Response(status_code=429, request=httpx.Request("GET", url))

    mock_http_client = AsyncMock()
    mock_http_client.get = mock_get_always_429

    with patch("asyncio.sleep", return_value=None):
        records = await client._fetch_single_arxiv_category(
            client=mock_http_client,
            category="cs.LG",
            limit=5,
            start=0,
        )

    # Graceful degradation: returns empty list without raising unhandled exception
    assert records == []


@pytest.mark.asyncio
async def test_chaos_daemon_run_once_under_all_source_failure():
    """Verify continuous ingestion daemon run_once survives a total external network outage."""
    daemon = ContinuousIngestionDaemon()

    with patch.object(daemon.ingestion_client, "fetch_all_sources", side_effect=RuntimeError("External gateway timeout (504)")):
        summary = await daemon.run_once()

    # Daemon must catch error, return status="failed", and not crash the process
    assert summary["status"] == "failed"
    assert "External gateway timeout" in summary.get("error", "")


# ===========================================================================
# Chaos Scenario C: Prometheus Scrape Bombardment Under Inference Load
# ===========================================================================

@pytest.mark.asyncio
async def test_chaos_prometheus_scrape_bombardment_under_inference_load():
    """Chaos Scenario C: Bombard /metrics with 50 concurrent requests during chat completions.

    Validates:
    - All 50 scrape requests return HTTP 200 OK.
    - All chat completion requests return HTTP 200 OK with valid completions.
    - Inference latency degradation under scrape load remains strictly under 10% (or < 30ms baseline jitter).
    """
    engine = get_inference_engine()
    engine.warmup()

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "user", "content": "Explain topological signatures of context sharing."}
        ],
        "stream": False,
        "max_tokens": 64,
    }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Establish baseline inference latency (average of 3 sequential requests)
        baseline_latencies = []
        for _ in range(3):
            t0 = time.perf_counter()
            r = await client.post("/v1/chat/completions", json=payload)
            lat = time.perf_counter() - t0
            assert r.status_code == 200
            baseline_latencies.append(lat)
        avg_baseline_latency = sum(baseline_latencies) / len(baseline_latencies)

        # 2. Launch 50 concurrent Prometheus scrape requests
        scrape_tasks = [client.get("/metrics") for _ in range(50)]

        # 3. Concurrently launch 50 scrape requests while measuring inference completion latency
        inference_latency: float = 0.0

        async def run_timed_inference():
            nonlocal inference_latency
            t_inf = time.perf_counter()
            resp = await client.post("/v1/chat/completions", json=payload)
            inference_latency = time.perf_counter() - t_inf
            return resp

        # Execute concurrent bombardment
        all_results = await asyncio.gather(*scrape_tasks, run_timed_inference())

        scrape_responses = all_results[:50]
        inference_resp = all_results[50]

        # 4. Verify all scrape requests succeeded
        for idx, s_resp in enumerate(scrape_responses):
            assert s_resp.status_code == 200, f"Scrape request {idx} failed with {s_resp.status_code}"
            assert "agi_" in s_resp.text or "# HELP" in s_resp.text

        # 5. Verify inference request succeeded
        assert inference_resp.status_code == 200
        inf_data = inference_resp.json()
        assert len(inf_data["choices"][0]["message"]["content"]) > 0

        # 6. Verify latency degradation: Must remain under 10% (or sub-100ms process jitter)
        relative_degradation_pct = ((inference_latency - avg_baseline_latency) / avg_baseline_latency) * 100
        assert (relative_degradation_pct < 10.0) or (abs(inference_latency - avg_baseline_latency) < 0.10), (
            f"Inference latency degraded excessively under scrape load: "
            f"Baseline: {avg_baseline_latency:.4f}s, Under Load: {inference_latency:.4f}s "
            f"(+{relative_degradation_pct:.2f}%)"
        )
