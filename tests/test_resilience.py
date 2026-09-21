"""Disaster Recovery & Self-Healing Resilience Test Suite.

Validates fault-tolerance and automated recovery for:
1. Mid-cycle MySQL connection drops during ReAct reasoning (in-memory buffering + auto-flush on reconnect).
2. Transient GCS network timeouts with Tenacity exponential backoff, jitter, and graceful degradation.
3. Corrupted LoRA adapter payloads during live hot-reload with atomic rollback to previous active model.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
import requests

from agent.orchestrator import ReActOrchestrator
from agent.tools import create_default_tool_registry
from config import Settings
from memory.vector_store import VectorStore
from serving.app import app, get_inference_engine
from storage.gcs_manager import GCSManager
from storage.mysql_manager import MySQLAuditManager


@pytest.fixture(autouse=True)
def clean_singletons():
    """Reset singletons before and after each test."""
    MySQLAuditManager.reset_singleton()
    yield
    MySQLAuditManager.reset_singleton()


# ===========================================================================
# 1. MySQL Dropped Connection & Self-Healing Telemetry Buffering
# ===========================================================================

def test_mysql_drop_mid_cycle_resilient_buffering_and_reconnect():
    """Simulate a dropped MySQL connection midway through a ReAct reasoning cycle.

    Verifies:
    - ReAct loop completes non-blocking without raising database exceptions.
    - Unwritten telemetry is captured in MySQLAuditManager in-memory buffer.
    - When connection is re-established, flush_buffer() retries and commits buffered logs.
    """
    settings = Settings(
        mysql_host="127.0.0.1",
        mysql_port=3306,
        mysql_database="agi_memory",
    )
    mysql_mgr = MySQLAuditManager(settings=settings)

    # Initially mock a connection failure midway through the cycle
    mysql_mgr._is_available = False
    mysql_mgr.get_connection = MagicMock(return_value=None)

    tools = create_default_tool_registry()

    # Initialize ReAct orchestrator with mocked audit manager
    orchestrator = ReActOrchestrator(
        tools=tools,
        max_steps=2,
        audit_manager=mysql_mgr,
    )

    # Execute ReAct cycle while MySQL is disconnected
    final_answer = orchestrator.run("What is the computational complexity of transformer self-attention?")

    # 1. Non-blocking verification: Agent execution must complete successfully
    assert final_answer is not None
    assert final_answer.success is True
    assert len(final_answer.steps) > 0

    # 2. In-Memory Buffering verification: Session audit log must be buffered safely
    assert mysql_mgr.buffered_count > 0, "Session telemetry must be buffered during database outage"
    assert len(mysql_mgr._buffered_sessions) == 1

    buffered_session, buffered_steps = mysql_mgr._buffered_sessions[0]
    assert "input_objective" in buffered_session
    assert buffered_session["status"] == "success"
    assert buffered_steps is not None
    assert len(buffered_steps) > 0

    # 3. Self-Healing Reconnect: Simulate MySQL restored and accepting writes
    mock_active_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_active_conn.cursor.return_value = mock_cursor

    # Restore get_connection to return active connection
    mysql_mgr.get_connection = MagicMock(return_value=mock_active_conn)

    # Trigger flush of in-memory telemetry buffer
    flush_results = mysql_mgr.flush_buffer()

    # 4. Verify flush succeeded and buffer is drained
    assert flush_results["flushed_sessions"] == 1
    assert mysql_mgr.buffered_count == 0
    assert mock_cursor.execute.call_count >= 1
    assert mock_active_conn.commit.called


# ===========================================================================
# 2. GCS Network Timeouts, Tenacity Backoff & Graceful Degradation
# ===========================================================================

def test_gcs_timeout_tenacity_backoff_and_graceful_degradation(tmp_path):
    """Simulate transient GCS network timeouts during checkpoint streaming and index sync.

    Verifies:
    - Tenacity retries transient timeouts up to configured max attempts with exponential backoff and jitter.
    - VectorStore.sync_to_gcs() handles persistent GCS outage gracefully without crashing.
    """
    settings = Settings(
        gcs_bucket_name="test-resilience-bucket",
        memory_local_dir=tmp_path / "memory",
    )

    test_file = tmp_path / "test_checkpoint.bin"
    test_file.write_bytes(b"dummy checkpoint data")

    # Track attempts made by tenacity
    attempt_count = 0

    def mock_upload_with_timeout(*args, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        raise requests.exceptions.Timeout("Connection timed out to storage.googleapis.com:443")

    # Mock the underlying blob.upload_from_file to trigger tenacity retries
    mock_blob = MagicMock()
    mock_blob.upload_from_file.side_effect = mock_upload_with_timeout

    mock_bucket = MagicMock()
    mock_bucket.blob.return_value = mock_blob

    mock_client = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    # Pass mock_client directly so GCSManager does not invoke ADC google.auth
    gcs_mgr = GCSManager(settings=settings, client=mock_client, force_new=True)

    # Fast tenacity execution: mock time.sleep during retry backoff so test is instantaneous
    with patch("time.sleep", return_value=None):
        with pytest.raises(requests.exceptions.Timeout):
            gcs_mgr.upload_file(test_file, "checkpoints/test_checkpoint.bin")

    assert attempt_count == 5, f"Tenacity should have retried 5 times, but got {attempt_count}"

    # 2. Verify graceful degradation during VectorStore.sync_to_gcs()
    vector_store = VectorStore(settings=settings, local_dir=tmp_path / "memory")
    vector_store.save_local()

    # Even if GCS fails completely, sync_to_gcs should degrade gracefully and report status
    with patch.object(gcs_mgr, "upload_file", side_effect=requests.exceptions.ConnectionError("GCS unreachable")):
        result = vector_store.sync_to_gcs(gcs_manager=gcs_mgr)
        assert result["status"] == "failed"
        assert "error" in result
        assert "GCS unreachable" in result["error"]


# ===========================================================================
# 3. Corrupted LoRA Adapter Payload & Atomic Rollback
# ===========================================================================

def test_corrupted_adapter_payload_atomic_rollback(tmp_path):
    """Simulate a failed/corrupted adapter payload in POST /v1/models/reload.

    Verifies:
    - Endpoint detects corrupted adapter metadata/weights and returns HTTP 500.
    - System rolls back active adapter to the previously active model.
    - FastAPI serving microservice remains healthy and continues serving completions.
    """
    client = TestClient(app)
    engine = get_inference_engine()
    engine.is_warmed_up = True

    # Establish initial known healthy adapter state
    engine.active_adapter_name = "stable_v1_adapter"
    engine.active_adapter_path = "/staging/adapters/stable_v1"
    engine.active_adapter_meta = {"version": "1.0.0", "weights_hash": "sha256:abc123healthy"}

    # Simulate corrupted download payload (invalid non-JSON config and empty safetensors)
    def fake_corrupt_download(remote_blob, local_path):
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if "adapter_config.json" in str(local_path):
            # Corrupted invalid JSON content
            target.write_text("<<<INVALID_JSON_CORRUPTED_PAYLOAD>>>", encoding="utf-8")
        else:
            # Corrupted 0-byte safetensors file
            target.write_bytes(b"")
        return target

    with patch("storage.gcs_manager.GCSManager.download_file", side_effect=fake_corrupt_download):
        response = client.post(
            "/v1/models/reload",
            json={"gcs_prefix": "models/lora_checkpoints/corrupted_run"},
        )

        # 1. Verify HTTP 500 error returned
        assert response.status_code == 500
        error_detail = response.json().get("detail", "")
        assert "corrupted adapter payload" in error_detail.lower()
        assert "rolled back to active adapter: stable_v1_adapter" in error_detail.lower()

        # 2. Verify state rolled back to previous active adapter
        assert engine.active_adapter_name == "stable_v1_adapter"
        assert engine.active_adapter_path == "/staging/adapters/stable_v1"
        assert engine.active_adapter_meta["version"] == "1.0.0"

        # 3. Verify healthcheck still reports healthy status
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        health_data = health_resp.json()
        assert health_data["status"] == "healthy"
        assert health_data["active_adapter"] == "stable_v1_adapter"

        # 4. Verify chat completions endpoint remains fully operational
        chat_resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "Verify system survivability after corrupted reload."}
                ],
                "stream": False,
            },
        )
        assert chat_resp.status_code == 200
        chat_data = chat_resp.json()
        assert len(chat_data["choices"]) > 0
        assert len(chat_data["choices"][0]["message"]["content"]) > 0
