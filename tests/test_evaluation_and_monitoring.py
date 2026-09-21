"""Automated Test Suite for Monitoring, Advanced Tooling, and A/B Evaluation.

Validates:
1. Prometheus metrics exposition, counters, histograms, and gauge updates.
2. Advanced tool sandboxing:
   - Shell allowlisting, timeout enforcement, and dangerous pattern rejection.
   - ArXiv PDF URL normalization and multi-page text extraction.
   - Read-only SQL query validation and injection rejection.
3. A/B evaluation benchmark execution, ranking, and report generation.
"""

from io import BytesIO
import json
from pathlib import Path
import re
import subprocess
import time
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from agent.tools import ToolRegistry
from agent.tools_advanced import (
    ALLOWED_COMMAND_BINARIES,
    ShellSecurityError,
    SQLSecurityError,
    create_extended_tool_registry,
    execute_shell_command,
    fetch_arxiv_pdf_text,
    normalize_arxiv_pdf_url,
    sql_query_executor,
    validate_read_only_sql,
    validate_shell_command,
)
from evaluation.ab_evaluator import (
    ABEvaluationHarness,
    BenchmarkScenario,
    EvaluationMetrics,
    compute_keyword_faithfulness,
    run_ab_evaluation,
)
from monitoring.metrics import (
    ACTIVE_MEMORY_VECTORS,
    FAISS_QUERY_LATENCY_SECONDS,
    MODEL_RELOAD_ATTEMPTS_TOTAL,
    MYSQL_BUFFER_QUEUE_SIZE,
    PAPERS_INGESTED_TOTAL,
    REACT_STEP_LATENCY_SECONDS,
    SYSTEM_RAM_USAGE_BYTES,
    SYSTEM_VRAM_USAGE_BYTES,
    TIME_TO_FIRST_TOKEN_SECONDS,
    TOKENS_PER_SECOND,
    TOOL_EXECUTION_CALLS_TOTAL,
    get_latest_metrics,
    record_model_reload,
    record_paper_ingested,
    record_tool_call,
    time_faiss_query,
    time_react_step,
    update_system_gauges,
)
from serving.app import app


# ===========================================================================
# 1. Prometheus Metrics Telemetry Tests
# ===========================================================================

def test_prometheus_metrics_endpoint():
    """Verify GET /metrics on serving microservice exposes valid Prometheus format."""
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    body = response.text
    # Verify core Prometheus metric families are registered
    assert "agi_papers_ingested_total" in body
    assert "agi_tool_execution_calls_total" in body
    assert "agi_model_reload_attempts_total" in body
    assert "agi_react_step_latency_seconds" in body
    assert "agi_time_to_first_token_seconds" in body
    assert "agi_tokens_per_second" in body
    assert "agi_faiss_query_latency_seconds" in body
    assert "agi_active_memory_vectors" in body
    assert "agi_mysql_buffer_queue_size" in body
    assert "agi_system_ram_usage_bytes" in body
    assert "agi_system_vram_usage_bytes" in body


def test_metric_collectors_and_timing_helpers():
    """Verify counter incrementation, gauge sampling, and histogram observation."""
    # Test counters
    record_paper_ingested("cs.AI", 3)
    record_paper_ingested("stat.ML", 2)
    record_tool_call("query_vector_memory", success=True)
    record_tool_call("execute_shell_command", success=False)
    record_model_reload("success")
    record_model_reload("corrupted_rollback")

    # Test gauges
    gauges = update_system_gauges(active_vectors=128, buffered_count=4)
    assert gauges["active_vectors"] == 128
    assert gauges["buffered_count"] == 4
    assert gauges["ram_bytes"] > 0

    # Test histograms via context managers
    with time_react_step():
        time.sleep(0.005)

    with time_faiss_query():
        time.sleep(0.002)

    metrics_text = get_latest_metrics().decode("utf-8")
    assert 'agi_papers_ingested_total{domain="cs.AI"}' in metrics_text
    assert 'agi_tool_execution_calls_total{status="success",tool_name="query_vector_memory"}' in metrics_text
    assert 'agi_tool_execution_calls_total{status="failure",tool_name="execute_shell_command"}' in metrics_text
    assert 'agi_model_reload_attempts_total{status="corrupted_rollback"}' in metrics_text
    assert 'agi_active_memory_vectors 128.0' in metrics_text
    assert 'agi_mysql_buffer_queue_size 4.0' in metrics_text


# ===========================================================================
# 2. Advanced Tool Sandboxing Tests
# ===========================================================================

def test_shell_command_allowlisting_and_execution():
    """Verify allowed shell commands execute and return stdout."""
    # Safe echo command
    res = execute_shell_command("echo Hello Sandboxed World")
    assert "Hello Sandboxed World" in res

    # Safe python command
    py_res = execute_shell_command('python -c "print(7 * 6)"')
    assert "42" in py_res


def test_shell_command_security_rejections():
    """Verify disallowed and destructive shell commands are blocked."""
    # Forbidden binaries
    with pytest.raises(ShellSecurityError) as exc_info:
        validate_shell_command("rm -rf /")
    assert "prohibited" in str(exc_info.value).lower() or "not in the sandbox allowlist" in str(exc_info.value).lower()

    # Banned commands should return security error string via execute_shell_command
    res1 = execute_shell_command("sudo apt-get update")
    assert "SECURITY ERROR" in res1

    res2 = execute_shell_command("del /f /s /q *.*")
    assert "SECURITY ERROR" in res2

    res3 = execute_shell_command("dd if=/dev/zero of=/dev/sda")
    assert "SECURITY ERROR" in res3

    # Chaining into unallowlisted utility
    res4 = execute_shell_command("echo test && whoami && netstat -an")
    assert "SECURITY ERROR" in res4
    assert "netstat" in res4


def test_shell_command_timeout_enforcement():
    """Verify long-running commands are terminated at configured timeout."""
    # Sleep 5 seconds with 1 second timeout
    start_t = time.perf_counter()
    res = execute_shell_command('python -c "import time; time.sleep(5)"', timeout_seconds=1)
    duration = time.perf_counter() - start_t

    assert "timed out after 1 seconds" in res.lower() or "timeout" in res.lower()
    assert duration < 4.0  # Must terminate promptly


def test_arxiv_pdf_url_normalization():
    """Verify ArXiv paper and abstract URLs normalize to direct PDF links."""
    assert normalize_arxiv_pdf_url("https://arxiv.org/abs/2301.07041") == "https://arxiv.org/pdf/2301.07041.pdf"
    assert normalize_arxiv_pdf_url("https://arxiv.org/abs/2303.08774v2") == "https://arxiv.org/pdf/2303.08774v2.pdf"
    assert normalize_arxiv_pdf_url("https://arxiv.org/pdf/2301.07041.pdf") == "https://arxiv.org/pdf/2301.07041.pdf"


def test_fetch_arxiv_pdf_text_extraction(tmp_path):
    """Verify pypdf parses multi-page PDF bytes into formatted text blocks."""
    # Create a synthetic 2-page PDF in-memory
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)

    pdf_buffer = BytesIO()
    writer.write(pdf_buffer)
    pdf_bytes = pdf_buffer.getvalue()

    # Mock httpx.Client to return our synthetic PDF
    mock_resp = MagicMock()
    mock_resp.content = pdf_bytes
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.get", return_value=mock_resp):
        res = fetch_arxiv_pdf_text("https://arxiv.org/abs/2301.99999", max_pages=2)
        assert "=== Extracted 2/2 Pages" in res
        assert "--- [Page 1 of 2] ---" in res
        assert "--- [Page 2 of 2] ---" in res


def test_read_only_sql_query_validation():
    """Verify read-only SQL validation allows queries and blocks destructive DDL/DML."""
    # Valid read-only queries
    assert validate_read_only_sql("SELECT id, title FROM research_papers") == "SELECT id, title FROM research_papers"
    assert validate_read_only_sql("SHOW TABLES") == "SHOW TABLES"
    assert validate_read_only_sql("DESCRIBE agent_sessions") == "DESCRIBE agent_sessions"
    assert validate_read_only_sql("EXPLAIN SELECT * FROM training_runs") == "EXPLAIN SELECT * FROM training_runs"

    # Reject modifying statements
    for banned in ["DROP TABLE research_papers", "DELETE FROM agent_sessions", "INSERT INTO training_runs VALUES (1)", "UPDATE research_papers SET title='x'", "TRUNCATE TABLE agent_sessions"]:
        with pytest.raises(SQLSecurityError):
            validate_read_only_sql(banned)

    # Reject multi-statement injection attempts
    with pytest.raises(SQLSecurityError):
        validate_read_only_sql("SELECT 1; DROP TABLE users")


def test_sql_query_executor_security_handling():
    """Verify sql_query_executor returns clean error diagnostics on violations."""
    res = sql_query_executor("DROP DATABASE agi_memory")
    assert "SECURITY ERROR" in res
    assert "permitted" in res or "prohibited" in res

    multi_res = sql_query_executor("SELECT * FROM research_papers; DELETE FROM research_papers")
    assert "SECURITY ERROR" in multi_res
    assert "prohibited" in multi_res


def test_extended_tool_registry_schemas():
    """Verify all 6 core and advanced tools conform to OpenAI/Anthropic registry schemas."""
    registry = create_extended_tool_registry(include_core=True)
    assert len(registry._tools) == 6
    expected_tools = {
        "query_vector_memory", "execute_python_code", "fetch_live_web",
        "execute_shell_command", "fetch_arxiv_pdf_text", "sql_query_executor"
    }
    assert set(registry._tools.keys()) == expected_tools

    # Test OpenAI schema generation
    openai_schemas = registry.get_openai_schemas()
    assert len(openai_schemas) == 6
    for s in openai_schemas:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "parameters" in s["function"]

    # Test Anthropic schema generation
    anthropic_schemas = registry.get_anthropic_schemas()
    assert len(anthropic_schemas) == 6
    for s in anthropic_schemas:
        assert "name" in s
        assert "input_schema" in s


# ===========================================================================
# 3. A/B Evaluation Benchmark Tests
# ===========================================================================

def test_keyword_faithfulness_metric():
    """Verify keyword presence calculation."""
    sample_text = "LoRA parameter-efficient fine-tuning reduces memory footprint and optimizes adapter weights."
    kws = ["lora", "parameter", "memory", "weights", "quantum"]
    score = compute_keyword_faithfulness(sample_text, kws)
    # 4 out of 5 matched = 80.0%
    assert score == 80.0


def test_ab_evaluator_harness_execution(tmp_path):
    """Verify full evaluation execution across 4 variants and generation of leaderboard and JSON."""
    test_scenarios = [
        BenchmarkScenario(
            scenario_id="TEST-01",
            category="Vector Search",
            prompt="Find papers on neural network quantization.",
            expected_tools=["query_vector_memory"],
            reference_keywords=["quantization", "neural", "network"],
        ),
        BenchmarkScenario(
            scenario_id="TEST-02",
            category="Code Execution",
            prompt="Calculate 2 to the power of 10 in python.",
            expected_tools=["execute_python_code"],
            reference_keywords=["1024", "power", "calculate"],
        ),
    ]

    harness = ABEvaluationHarness(scenarios=test_scenarios)
    results = harness.run_full_evaluation()

    assert len(results) == 4
    # All variants should have computed valid metrics
    for r in results:
        assert r.metrics.total_tasks == 2
        assert 0.0 <= r.metrics.composite_score <= 100.0
        assert len(r.task_details) == 2

    # Verify leaderboard markdown generation
    md_report = harness.generate_leaderboard_markdown(results)
    assert "# LLM Backbone & Reasoning Architecture Evaluation Leaderboard" in md_report
    assert "Overall Variant Rankings" in md_report
    assert "Variant A" in md_report
    assert "Variant D" in md_report

    # Test run_ab_evaluation output persistence
    summary = run_ab_evaluation(output_dir=tmp_path, scenarios=test_scenarios)
    assert Path(summary["leaderboard_path"]).exists()
    assert Path(summary["json_path"]).exists()

    with open(summary["json_path"], "r", encoding="utf-8") as f:
        json_data = json.load(f)
    assert json_data["total_variants_evaluated"] == 4
    assert len(json_data["rankings"]) == 4
