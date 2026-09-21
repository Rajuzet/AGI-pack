"""Integration and unit tests for MySQLAuditManager, relational schema, and offline fallback."""

import json
import os
from pathlib import Path
import time
from unittest.mock import MagicMock, patch
import pytest

from config import Settings
from storage.mysql_manager import MySQLAuditManager, get_mysql_manager


@pytest.fixture(autouse=True)
def reset_mysql_singleton():
    """Ensure MySQL manager singleton state is reset cleanly before and after tests."""
    MySQLAuditManager.reset_singleton()
    yield
    MySQLAuditManager.reset_singleton()


def test_mysql_singleton_behavior():
    """Verify singleton pattern ensures a single pool/manager instance."""
    settings = Settings(mysql_host="127.0.0.1", mysql_port=3306, mysql_database="test_db")
    mgr1 = MySQLAuditManager(settings=settings)
    mgr2 = MySQLAuditManager()
    mgr3 = get_mysql_manager()

    assert mgr1 is mgr2
    assert mgr2 is mgr3
    assert mgr1.host == "127.0.0.1"
    assert mgr1.database == "test_db"


def test_schema_sql_definitions():
    """Verify storage/schema.sql contains required tables, utf8mb4 encoding, and FK cascade."""
    schema_path = Path(__file__).resolve().parent.parent / "storage" / "schema.sql"
    assert schema_path.exists(), "storage/schema.sql must exist"

    sql_content = schema_path.read_text(encoding="utf-8")
    assert "agi_memory" in sql_content
    assert "utf8mb4" in sql_content
    assert "CREATE TABLE IF NOT EXISTS research_papers" in sql_content
    assert "CREATE TABLE IF NOT EXISTS training_runs" in sql_content
    assert "CREATE TABLE IF NOT EXISTS agent_sessions" in sql_content
    assert "CREATE TABLE IF NOT EXISTS agent_steps" in sql_content

    # Check foreign key with cascading delete
    assert "FOREIGN KEY (session_id)" in sql_content
    assert "REFERENCES agent_sessions(session_id)" in sql_content
    assert "ON DELETE CASCADE" in sql_content

    # Check indexes
    assert "idx_papers_source" in sql_content
    assert "idx_papers_ingested_at" in sql_content


def test_offline_fallback_non_blocking():
    """Verify that when MySQL is offline/unreachable, all operations degrade gracefully."""
    settings = Settings(
        mysql_host="192.0.2.1",  # Non-routable test address
        mysql_port=3306,
        mysql_database="test_agi",
    )
    with patch("mysql.connector.pooling.MySQLConnectionPool", side_effect=Exception("Connection refused")):
        mgr = MySQLAuditManager(settings=settings)

        # 1. Connection acquisition returns None without raising
        conn = mgr.get_connection()
        assert conn is None

        # 2. Health check reports unreachable / degraded
        health = mgr.health_check()
        assert health["healthy"] is False
        assert health["status"] in ("unreachable", "degraded")
        assert "Connection refused" in health["error"]

        # 3. Operations return False without raising exceptions
        assert mgr.record_paper({"id": "p1", "title": "Test Paper"}) is False
        assert mgr.record_training_run({"run_id": "r1", "base_model": "Qwen"}) is False
        assert mgr.record_agent_session({"session_id": "s1"}, [{"step_index": 1}]) is False
        assert mgr.init_schema() is False


def test_record_paper_successful_execution():
    """Verify record_paper properly serializes JSON metadata and executes parameterized SQL."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_pool.get_connection.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    with patch("mysql.connector.pooling.MySQLConnectionPool", return_value=mock_pool):
        mgr = MySQLAuditManager(settings=Settings())
        paper_payload = {
            "id": "arxiv:2401.12345",
            "title": "Autonomous LLM Reasoning with Free GPU Checkpoints",
            "abstract": "We demonstrate QLoRA fine-tuning and relational audit logging.",
            "source": "arxiv",
            "metadata": {
                "authors": ["Alice Smith", "Bob Jones"],
                "categories": ["cs.AI", "cs.LG"],
                "pdf_url": "https://arxiv.org/pdf/2401.12345.pdf",
            },
        }

        success = mgr.record_paper(paper_payload)
        assert success is True

        # Verify cursor execution
        assert mock_cursor.execute.called
        sql_arg, params_arg = mock_cursor.execute.call_args[0]
        assert "INSERT INTO research_papers" in sql_arg
        assert params_arg[0] == "arxiv:2401.12345"
        assert params_arg[1] == "Autonomous LLM Reasoning with Free GPU Checkpoints"
        assert params_arg[3] == "arxiv"

        # Check JSON serialization of authors and categories
        authors_deserialized = json.loads(params_arg[4])
        assert authors_deserialized == ["Alice Smith", "Bob Jones"]
        cats_deserialized = json.loads(params_arg[5])
        assert cats_deserialized == ["cs.AI", "cs.LG"]
        assert params_arg[6] == "https://arxiv.org/pdf/2401.12345.pdf"

        # Verify commit and close
        assert mock_conn.commit.called
        assert mock_cursor.close.called
        assert mock_conn.close.called


def test_record_training_run_successful_execution():
    """Verify record_training_run properly inserts run metrics and GCS checkpoint URI."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_pool.get_connection.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    with patch("mysql.connector.pooling.MySQLConnectionPool", return_value=mock_pool):
        mgr = MySQLAuditManager(settings=Settings())
        run_payload = {
            "run_id": "run_lora_20260920",
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "quantization": "4-bit NF4",
            "training_samples": 42,
            "training_loss": 0.385,
            "duration_seconds": 124.5,
            "gcs_sync": {
                "target_uri": "gs://autonomous-agi-data-pack-primary/checkpoints/run_lora_20260920"
            },
        }

        success = mgr.record_training_run(run_payload)
        assert success is True

        assert mock_cursor.execute.called
        sql_arg, params_arg = mock_cursor.execute.call_args[0]
        assert "INSERT INTO training_runs" in sql_arg
        assert params_arg[0] == "run_lora_20260920"
        assert params_arg[1] == "Qwen/Qwen2.5-7B-Instruct"
        assert params_arg[2] == "4-bit NF4"
        assert params_arg[3] == 42
        assert params_arg[4] == 0.385
        assert params_arg[5] == 124.5
        assert params_arg[6] == "gs://autonomous-agi-data-pack-primary/checkpoints/run_lora_20260920"

        assert mock_conn.commit.called
        assert mock_cursor.close.called
        assert mock_conn.close.called


def test_record_agent_session_with_steps_and_foreign_keys():
    """Verify record_agent_session inserts session record and linked step records in one transaction."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_pool.get_connection.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    with patch("mysql.connector.pooling.MySQLConnectionPool", return_value=mock_pool):
        mgr = MySQLAuditManager(settings=Settings())
        session_data = {
            "session_id": "sess_abc123",
            "input_objective": "Calculate Fibonacci sequence and verify in memory",
            "status": "success",
            "total_steps": 2,
            "retry_count": 1,
            "execution_latency": 1.45,
        }
        steps = [
            {
                "step_index": 1,
                "thought_trace": "Need to execute python snippet",
                "action_name": "execute_python_code",
                "action_input": {"code": "print(fib(10))"},
                "observation_output": "NameError: name 'fib' is not defined",
                "step_latency": 0.45,
            },
            {
                "step_index": 2,
                "thought_trace": "Define fib function before calling it",
                "action_name": "execute_python_code",
                "action_input": {"code": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)\nprint(fib(10))"},
                "observation_output": "55",
                "step_latency": 0.55,
            },
        ]

        success = mgr.record_agent_session(session_data, steps)
        assert success is True

        # Check that cursor was executed 3 times (1 session + 2 steps)
        assert mock_cursor.execute.call_count == 3

        # First call: agent_sessions
        first_call = mock_cursor.execute.call_args_list[0]
        assert "INSERT INTO agent_sessions" in first_call[0][0]
        assert first_call[0][1][0] == "sess_abc123"
        assert first_call[0][1][1] == "Calculate Fibonacci sequence and verify in memory"
        assert first_call[0][1][2] == "success"

        # Second call: agent_steps step 1 (referencing session_id)
        second_call = mock_cursor.execute.call_args_list[1]
        assert "INSERT INTO agent_steps" in second_call[0][0]
        assert second_call[0][1][0] == "sess_abc123"
        assert second_call[0][1][1] == 1
        assert second_call[0][1][3] == "execute_python_code"

        # Third call: agent_steps step 2
        third_call = mock_cursor.execute.call_args_list[2]
        assert third_call[0][1][0] == "sess_abc123"
        assert third_call[0][1][1] == 2

        # Verify transaction commit
        assert mock_conn.commit.called
        assert mock_cursor.close.called
        assert mock_conn.close.called


def test_health_check_reporting_metrics():
    """Verify health_check calculates latency and reports healthy status when alive."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_pool.get_connection.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    with patch("mysql.connector.pooling.MySQLConnectionPool", return_value=mock_pool):
        settings = Settings(mysql_host="db.internal.net", mysql_port=3306, mysql_database="prod_db")
        mgr = MySQLAuditManager(settings=settings)

        diag = mgr.health_check()
        assert diag["healthy"] is True
        assert diag["status"] == "healthy"
        assert diag["host"] == "db.internal.net"
        assert diag["port"] == 3306
        assert diag["database"] == "prod_db"
        assert diag["latency_ms"] >= 0.0
        assert diag["error"] is None


def test_react_orchestrator_session_audit_integration():
    """Verify that ReActOrchestrator automatically logs completed sessions to MySQLAuditManager."""
    from agent.orchestrator import ReActOrchestrator

    mock_audit_mgr = MagicMock()
    mock_audit_mgr.record_agent_session.return_value = True

    orchestrator = ReActOrchestrator(audit_manager=mock_audit_mgr)
    result = orchestrator.run("What is 12 * 12?")

    assert result.success is True
    # Audit manager must have recorded the session
    assert mock_audit_mgr.record_agent_session.called
    session_arg, steps_arg = mock_audit_mgr.record_agent_session.call_args[0]
    assert session_arg["status"] == "success"
    assert session_arg["input_objective"] == "What is 12 * 12?"
    assert len(steps_arg) == result.total_steps
