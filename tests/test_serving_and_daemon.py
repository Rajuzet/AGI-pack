"""Unit and integration tests for FastAPI serving microservice, adapter hot-reloading, and continuous ingestion daemon."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from config import Settings
from daemon.continuous_ingest import ContinuousIngestionDaemon
from ingestion.stream_sources import StandardizedRecord
from serving.app import app, get_inference_engine


@pytest.fixture
def client():
    """FastAPI TestClient with warmed up inference engine."""
    engine = get_inference_engine()
    engine.warmup()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Verify /health reports model warm-up status, backend, and VRAM telemetry."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ("healthy", "warming_up")
    assert data["warmed_up"] is True
    assert "base_model" in data
    assert "backend" in data
    assert "device" in data
    assert "vram_allocated_gb" in data
    assert "vram_reserved_gb" in data


def test_chat_completions_non_streaming(client):
    """Verify /v1/chat/completions returns OpenAI-compliant JSON response."""
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 15 * 15?"},
        ],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 256,
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["object"] == "chat.completion"
    assert data["id"].startswith("chatcmpl-")
    assert len(data["choices"]) == 1
    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert len(choice["message"]["content"]) > 0
    assert choice["finish_reason"] == "stop"

    usage = data["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_completions_streaming(client):
    """Verify /v1/chat/completions with stream=True returns Server-Sent Events ending in [DONE]."""
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "user", "content": "Explain quantum superposition briefly."},
        ],
        "stream": True,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = response.text.strip().split("\n\n")
    assert len(lines) >= 2

    # Check SSE structure of data lines
    data_lines = [l for l in lines if l.startswith("data: ")]
    assert len(data_lines) >= 2

    # Verify first chunk JSON schema
    first_chunk_raw = data_lines[0].replace("data: ", "")
    first_chunk = json.loads(first_chunk_raw)
    assert first_chunk["object"] == "chat.completion.chunk"
    assert "choices" in first_chunk
    assert "delta" in first_chunk["choices"][0]

    # Verify last SSE line is the OpenAI terminal sentinel
    assert data_lines[-1] == "data: [DONE]"


def test_hot_reload_adapter_success(client, tmp_path):
    """Verify /v1/models/reload hot-swaps PEFT adapter weights downloaded from GCS."""
    def fake_download(remote_blob, local_path):
        target = Path(local_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if "adapter_config.json" in str(local_path):
            target.write_text(json.dumps({"base_model": "Qwen/Qwen2.5-7B-Instruct", "peft_type": "LORA"}))
        else:
            target.write_text("mock weights content")
        return target

    with patch("storage.gcs_manager.GCSManager.download_file", side_effect=fake_download):
        response = client.post(
            "/v1/models/reload",
            json={"gcs_prefix": "models/lora_checkpoints/run_test"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["active_adapter"] == "latest"
        assert data["gcs_prefix"] == "models/lora_checkpoints/run_test"


def test_hot_reload_adapter_missing_gcs_checkpoint(client):
    """Verify /v1/models/reload returns 404 when adapter weights are missing in GCS."""
    with patch("storage.gcs_manager.GCSManager.download_file", side_effect=FileNotFoundError("Blob not found in bucket")):
        response = client.post(
            "/v1/models/reload",
            json={"gcs_prefix": "models/lora_checkpoints/non_existent"},
        )
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


@pytest.mark.asyncio
async def test_daemon_deduplication_against_mysql(tmp_path):
    """Verify continuous ingestion daemon skips records already present in MySQL."""
    settings = Settings(
        staging_dir=tmp_path / "staging",
        memory_local_dir=tmp_path / "memory",
    )
    settings.staging_dir.mkdir(parents=True, exist_ok=True)
    settings.memory_local_dir.mkdir(parents=True, exist_ok=True)

    daemon = ContinuousIngestionDaemon(settings=settings)

    mock_records = [
        StandardizedRecord(
            id="arxiv:2401.0001",
            title="Old Existing Paper",
            content="Already processed",
            source="arxiv",
            timestamp="2026-09-01T00:00:00Z",
        ),
        StandardizedRecord(
            id="arxiv:2401.0002",
            title="Brand New Paper",
            content="Novel attention mechanism with empirical proofs",
            source="arxiv",
            timestamp="2026-09-02T00:00:00Z",
        ),
    ]

    daemon.ingestion_client.fetch_all_sources = AsyncMock(return_value=mock_records)
    # MySQL reports that arxiv:2401.0001 is already present
    daemon.mysql_mgr.get_existing_paper_ids = MagicMock(return_value={"arxiv:2401.0001"})
    daemon.mysql_mgr.record_paper = MagicMock(return_value=True)
    daemon.gcs_mgr.upload_file = MagicMock(return_value="gs://bucket/blob")
    daemon.vector_store.sync_to_gcs = MagicMock(return_value={"status": "uploaded"})

    summary = await daemon.run_once()

    assert summary["status"] == "success"
    assert summary["harvested"] == 2
    assert summary["new_indexed"] == 1
    assert daemon.stats["total_deduped"] == 1

    # Verify only the new record was audited
    assert daemon.mysql_mgr.record_paper.call_count == 1
    audited_paper = daemon.mysql_mgr.record_paper.call_args[0][0]
    assert audited_paper["id"] == "arxiv:2401.0002"


@pytest.mark.asyncio
async def test_daemon_all_records_deduplicated(tmp_path):
    """Verify daemon skips indexing and GCS sync when all candidate records already exist."""
    settings = Settings(
        staging_dir=tmp_path / "staging",
        memory_local_dir=tmp_path / "memory",
    )
    daemon = ContinuousIngestionDaemon(settings=settings)

    mock_records = [
        StandardizedRecord(
            id="arxiv:111",
            title="Existing 1",
            content="Content 1",
            source="arxiv",
            timestamp="2026-09-01T00:00:00Z",
        )
    ]
    daemon.ingestion_client.fetch_all_sources = AsyncMock(return_value=mock_records)
    daemon.mysql_mgr.get_existing_paper_ids = MagicMock(return_value={"arxiv:111"})
    daemon.gcs_mgr.upload_file = MagicMock()

    summary = await daemon.run_once()

    assert summary["status"] == "deduped"
    assert summary["harvested"] == 1
    assert summary["new_indexed"] == 0
    # No GCS upload triggered since no new records
    assert not daemon.gcs_mgr.upload_file.called


def test_daemon_stop_signal_handling():
    """Verify daemon stop() properly triggers shutdown event."""
    daemon = ContinuousIngestionDaemon()
    daemon.is_running = True
    assert not daemon._stop_event.is_set()

    daemon.stop()
    assert daemon.is_running is False
    assert daemon._stop_event.is_set()


def test_orchestrator_live_serving_hook_integration():
    """Verify ReActOrchestrator successfully queries local serving endpoint."""
    from agent.orchestrator import ReActOrchestrator

    mock_serving_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Thought: I will compute the answer directly.\nFinal Answer: 42 is the answer.",
                }
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_serving_response
    mock_resp.raise_for_status = MagicMock()

    with patch("httpx.Client.post", return_value=mock_resp) as mock_post:
        orchestrator = ReActOrchestrator(
            serving_endpoint="http://localhost:8000/v1/chat/completions"
        )
        result = orchestrator.run("What is the meaning of life?")

        assert result.success is True
        assert "42 is the answer" in result.final_answer
        assert mock_post.called
        call_args, call_kwargs = mock_post.call_args
        assert call_args[0] == "http://localhost:8000/v1/chat/completions"
        assert "messages" in call_kwargs["json"]
