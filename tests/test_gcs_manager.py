"""Unit tests for GCSManager singleton, retry resiliency, chunked streaming, and sync."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from google.api_core.exceptions import GoogleAPICallError, NotFound, ServiceUnavailable
import requests.exceptions

from config import Settings
from storage.gcs_manager import (
    GCSManager,
    get_gcs_manager,
    is_transient_gcs_error,
    retry_gcs_operation,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure singleton state is cleanly reset before and after each test."""
    GCSManager.reset_singleton()
    yield
    GCSManager.reset_singleton()


def test_singleton_behavior():
    """Ensure multiple instantiations yield the exact same instance."""
    settings = Settings(gcs_bucket_name="test-singleton-bucket")
    mgr1 = GCSManager(settings=settings)
    mgr2 = GCSManager()
    mgr3 = get_gcs_manager()

    assert mgr1 is mgr2
    assert mgr2 is mgr3
    assert mgr1.bucket_name == "test-singleton-bucket"


def test_is_transient_gcs_error_classification():
    """Verify transient error classification matches retry policy."""
    # Transient errors
    assert is_transient_gcs_error(ServiceUnavailable("Backend service down"))
    assert is_transient_gcs_error(GoogleAPICallError("Too many requests", response=MagicMock(status_code=429)))
    assert is_transient_gcs_error(GoogleAPICallError("Internal server error", response=MagicMock(status_code=500)))
    assert is_transient_gcs_error(requests.exceptions.ConnectionError("Connection aborted"))
    assert is_transient_gcs_error(TimeoutError("Operation timed out"))
    assert is_transient_gcs_error(ConnectionResetError("Reset by peer"))

    # Fatal / non-transient errors
    assert not is_transient_gcs_error(NotFound("Object not found"))
    assert not is_transient_gcs_error(ValueError("Invalid argument"))
    assert not is_transient_gcs_error(PermissionError("Access denied"))


def test_retry_on_transient_error():
    """Verify tenacity retry decorator retries transient errors and succeeds."""
    attempts = 0

    @retry_gcs_operation
    def flaking_operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ServiceUnavailable("Temporary glitch")
        return "success"

    result = flaking_operation()
    assert result == "success"
    assert attempts == 3


def test_streaming_upload_file(tmp_path):
    """Verify chunked streaming upload configures chunk size and calls upload_from_file."""
    test_file = tmp_path / "large_dataset.bin"
    # Create a 512 KiB test file
    payload = b"X" * (512 * 1024)
    test_file.write_bytes(payload)

    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    settings = Settings(gcs_bucket_name="test-upload-bucket", gcs_chunk_size=256 * 1024)
    mgr = GCSManager(settings=settings, client=mock_client, force_new=True)

    uri = mgr.upload_file(test_file, "remote/large_dataset.bin", content_type="application/octet-stream")

    assert uri == "gs://test-upload-bucket/remote/large_dataset.bin"
    mock_bucket.blob.assert_called_once_with("remote/large_dataset.bin", chunk_size=256 * 1024)
    mock_blob.upload_from_file.assert_called_once()
    # Verify stream size parameter passed accurately
    _, kwargs = mock_blob.upload_from_file.call_args
    assert kwargs.get("size") == len(payload)
    assert kwargs.get("content_type") == "application/octet-stream"


def test_streaming_download_file(tmp_path):
    """Verify streaming download checks existence, writes to atomic temp, and renames."""
    dest_file = tmp_path / "downloaded" / "output.bin"

    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.exists.return_value = True

    # Simulate writing data inside download_to_file
    def fake_download(file_obj):
        file_obj.write(b"downloaded stream bytes")

    mock_blob.download_to_file.side_effect = fake_download
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    settings = Settings(gcs_bucket_name="test-dl-bucket")
    mgr = GCSManager(settings=settings, client=mock_client, force_new=True)

    result_path = mgr.download_file("remote/data.bin", dest_file)

    assert result_path == dest_file
    assert dest_file.exists()
    assert dest_file.read_bytes() == b"downloaded stream bytes"


def test_sync_directory_to_bucket(tmp_path):
    """Verify directory synchronization uploads missing/changed files."""
    sync_dir = tmp_path / "sync_source"
    sync_dir.mkdir()
    (sync_dir / "file1.txt").write_text("Hello file 1")
    sub = sync_dir / "subdir"
    sub.mkdir()
    (sub / "file2.txt").write_text("Hello file 2")

    mock_client = MagicMock()
    mock_bucket = MagicMock()
    # Mock blob as non-existent initially to trigger upload
    mock_bucket.get_blob.return_value = None
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_client.bucket.return_value = mock_bucket

    settings = Settings(gcs_bucket_name="test-sync-bucket")
    mgr = GCSManager(settings=settings, client=mock_client, force_new=True)

    summary = mgr.sync_directory_to_bucket(sync_dir, remote_prefix="backup")

    assert summary["uploaded_count"] == 2
    assert summary["skipped_count"] == 0
    assert summary["failed_count"] == 0
    assert "file1.txt" in summary["uploaded_files"]
    assert "subdir/file2.txt" in summary["uploaded_files"]


def test_bucket_health_check_healthy():
    """Verify health check returns healthy status when bucket exists."""
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_bucket.location = "US-CENTRAL1"
    mock_bucket.storage_class = "STANDARD"
    mock_client.get_bucket.return_value = mock_bucket

    settings = Settings(gcs_bucket_name="prod-agi-bucket")
    mgr = GCSManager(settings=settings, client=mock_client, force_new=True)

    result = mgr.bucket_health_check()
    assert result["status"] == "healthy"
    assert result["exists"] is True
    assert result["location"] == "US-CENTRAL1"
    assert result["storage_class"] == "STANDARD"
    assert result["latency_ms"] >= 0


def test_bucket_health_check_not_found():
    """Verify health check gracefully captures NotFound errors."""
    mock_client = MagicMock()
    mock_client.get_bucket.side_effect = NotFound("Bucket not found")

    settings = Settings(gcs_bucket_name="non-existent-bucket")
    mgr = GCSManager(settings=settings, client=mock_client, force_new=True)

    result = mgr.bucket_health_check()
    assert result["status"] == "unhealthy"
    assert result["exists"] is False
    assert "not found" in result["error"].lower()


def test_gcs_credentials_path_resolution(tmp_path, monkeypatch):
    """Verify Settings resolves credentials from both GCS_CREDENTIALS_PATH and GOOGLE_APPLICATION_CREDENTIALS."""
    fake_key1 = tmp_path / "key1.json"
    fake_key1.write_text('{"type": "service_account"}')
    fake_key2 = tmp_path / "key2.json"
    fake_key2.write_text('{"type": "service_account"}')

    # Priority 1: GCS_CREDENTIALS_PATH directly
    settings1 = Settings(gcs_credentials_path=str(fake_key1))
    assert settings1.gcs_credentials_path == str(fake_key1)

    # Priority 2: GOOGLE_APPLICATION_CREDENTIALS from environment
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(fake_key2))
    settings2 = Settings(gcs_credentials_path=None)
    assert settings2.gcs_credentials_path == str(fake_key2)


def test_gcs_manager_local_fallback_no_credentials(tmp_path, monkeypatch, caplog):
    """Verify GCSManager gracefully falls back to local storage lake when no credentials exist."""
    monkeypatch.delenv("GCS_CREDENTIALS_PATH", raising=False)
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)

    lake_dir = tmp_path / "local_lake"
    settings = Settings(
        gcs_bucket_name="fallback-test-bucket",
        gcs_credentials_path=None,
        local_lake_dir=lake_dir,
    )

    mgr = GCSManager(settings=settings, force_new=True)

    # 1. Health check returns local fallback
    health = mgr.bucket_health_check()
    assert health["status"] == "local_fallback"
    assert health["healthy"] is True
    assert health["exists"] is True
    assert health["location"] == "local_disk"
    assert "data_storage/local_lake" in health["message"] or "local disk fallback mode" in health["message"]

    # 2. Upload file persists to local lake
    test_src = tmp_path / "sample.txt"
    test_src.write_text("Local lake fallback payload")
    uri = mgr.upload_file(test_src, "harvested/sample.txt")
    assert uri == "gs://fallback-test-bucket/harvested/sample.txt"
    assert (lake_dir / "harvested" / "sample.txt").exists()
    assert (lake_dir / "harvested" / "sample.txt").read_text() == "Local lake fallback payload"

    # 3. Download file retrieves from local lake
    download_dest = tmp_path / "downloaded_sample.txt"
    mgr.download_file("harvested/sample.txt", download_dest)
    assert download_dest.exists()
    assert download_dest.read_text() == "Local lake fallback payload"

    # 4. Sync directory to local lake
    sync_src = tmp_path / "sync_folder"
    sync_src.mkdir()
    (sync_src / "doc1.json").write_text('{"doc": 1}')
    (sync_src / "doc2.json").write_text('{"doc": 2}')

    summary = mgr.sync_directory_to_bucket(sync_src, remote_prefix="staged_sync")
    assert summary["uploaded_count"] == 2
    assert summary["mode"] == "local_disk_fallback"
    assert (lake_dir / "staged_sync" / "doc1.json").exists()
    assert (lake_dir / "staged_sync" / "doc2.json").exists()


def test_flush_local_lake_to_gcs_fallback_and_authenticated(tmp_path):
    """Verify flush_local_lake_to_gcs handles unauthenticated and authenticated modes."""
    lake_dir = tmp_path / "lake"
    lake_dir.mkdir(parents=True, exist_ok=True)
    mem_dir = tmp_path / "mem"
    mem_dir.mkdir(parents=True, exist_ok=True)

    (lake_dir / "doc.json").write_text('{"flushed": true}')
    (mem_dir / "index.faiss").write_bytes(b"faiss_bytes")
    (mem_dir / "metadata.json").write_text('{"total": 1}')

    settings = Settings(
        local_lake_dir=lake_dir,
        memory_local_dir=mem_dir,
        gcs_credentials_path=None,
    )

    # 1. Test unauthenticated fallback mode
    mgr_unauth = GCSManager(settings=settings, force_new=True)
    res_unauth = mgr_unauth.flush_local_lake_to_gcs()
    assert res_unauth["status"] == "skipped_unauthenticated"
    assert res_unauth["flushed_count"] == 0

    # 2. Test authenticated mode with mock GCS client
    mock_client = MagicMock()
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket

    mgr_auth = GCSManager(settings=settings, client=mock_client, force_new=True)
    res_auth = mgr_auth.flush_local_lake_to_gcs(delete_after_upload=False, sync_memory=True)

    assert res_auth["status"] == "flushed"
    assert res_auth["flushed_count"] >= 2
    assert "doc.json" in res_auth["lake_files_flushed"]
    assert "memory/index.faiss" in res_auth["memory_files_flushed"]

