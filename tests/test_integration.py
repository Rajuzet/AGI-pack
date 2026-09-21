"""End-to-end integration test for ingestion, staging, and GCS chunked persistence."""

import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from config import Settings
from ingestion.stream_sources import StreamIngestionClient
from storage.gcs_manager import GCSManager
from tests.test_stream_sources import SAMPLE_ARXIV_XML, SAMPLE_RSS_XML


@pytest.fixture(autouse=True)
def reset_gcs():
    GCSManager.reset_singleton()
    yield
    GCSManager.reset_singleton()


@pytest.mark.asyncio
async def test_end_to_end_pipeline(tmp_path):
    """Verify ingestion -> staging JSONL -> GCS streaming upload pipeline."""
    staging_dir = tmp_path / "staging"
    settings = Settings(
        gcs_bucket_name="integration-test-bucket",
        staging_dir=staging_dir,
        gcs_chunk_size=256 * 1024,
    )

    # 1. Mock HTTP Client returning both ArXiv and RSS
    mock_http = AsyncMock()
    mock_http.__aenter__.return_value = mock_http
    mock_http.__aexit__.return_value = None

    async def fake_get(url):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "arxiv" in str(url):
            resp.text = SAMPLE_ARXIV_XML
        else:
            resp.text = SAMPLE_RSS_XML
        return resp

    mock_http.get.side_effect = fake_get

    # 2. Run ingestion client
    ingestion_client = StreamIngestionClient(settings=settings, client=mock_http)
    records = await ingestion_client.fetch_all_sources(arxiv_limit=5, rss_limit=5)
    assert len(records) >= 2

    # 3. Stage to JSONL
    staged_file = staging_dir / "pipeline_test.jsonl"
    ingestion_client.save_records_to_jsonl(records, staged_file)
    assert staged_file.exists()
    assert staged_file.stat().st_size > 0

    # 4. GCS Upload verification
    mock_gcs_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_gcs_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    gcs_mgr = GCSManager(settings=settings, client=mock_gcs_client, force_new=True)
    uri = gcs_mgr.upload_file(staged_file, "pipeline/output.jsonl", content_type="application/x-ndjson")

    assert uri == "gs://integration-test-bucket/pipeline/output.jsonl"
    mock_bucket.blob.assert_called_once_with("pipeline/output.jsonl", chunk_size=256 * 1024)
    mock_blob.upload_from_file.assert_called_once()
