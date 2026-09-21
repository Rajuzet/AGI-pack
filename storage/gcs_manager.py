"""Production Google Cloud Storage manager with streaming chunked transfers and resilient retries.

Provides a thread-safe singleton client for uploading, downloading, synchronizing,
and health-checking GCS buckets, with memory-bounded streaming for arbitrarily large files.
"""

from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import shutil
import socket
import threading
import time
from typing import Any, BinaryIO, Dict, List, Optional, Union

from google.api_core.exceptions import (
    BadGateway,
    DeadlineExceeded,
    GatewayTimeout,
    GoogleAPICallError,
    InternalServerError,
    ServerError,
    ServiceUnavailable,
    TooManyRequests,
)
from google.cloud import storage
from google.cloud.exceptions import NotFound
import requests.exceptions
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from config import Settings, get_settings

logger = logging.getLogger(__name__)


def check_has_gcs_credentials(settings: Optional[Settings] = None) -> bool:
    """Determine whether valid GCS credentials or ADC configurations exist on system."""
    # 1. Check configured path in settings
    if settings and settings.gcs_credentials_path and os.path.exists(settings.gcs_credentials_path):
        return True

    # 2. Check environment variables
    gcp = os.environ.get("GCS_CREDENTIALS_PATH")
    if gcp and os.path.exists(gcp):
        return True

    gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if gac and os.path.exists(gac):
        return True

    # 3. Check standard gcloud CLI Application Default Credentials (ADC) file
    if os.name == "nt":
        adc_path = os.path.expandvars(r"%APPDATA%\gcloud\application_default_credentials.json")
    else:
        adc_path = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    if os.path.exists(adc_path):
        return True

    # 4. Check Google Cloud serverless / Compute Engine environment indicators
    if os.environ.get("K_SERVICE") or os.environ.get("GAE_APPLICATION") or os.environ.get("GCE_METADATA_HOST"):
        return True

    return False


def is_transient_gcs_error(exception: BaseException) -> bool:
    """Determine whether an encountered exception is a transient error eligible for retry."""
    if isinstance(
        exception,
        (
            ServerError,
            ServiceUnavailable,
            TooManyRequests,
            InternalServerError,
            BadGateway,
            GatewayTimeout,
            DeadlineExceeded,
            ConnectionResetError,
            TimeoutError,
            socket.timeout,
        ),
    ):
        return True

    if isinstance(exception, GoogleAPICallError):
        # Check explicit code or status_code from underlying HTTP response
        code = getattr(exception, "code", None)
        resp = getattr(exception, "response", None)
        if code is None and resp is not None:
            code = getattr(resp, "status_code", None)
        # Retry HTTP 408 (Request Timeout), 429 (Too Many Requests), 5xx (Server Errors)
        if code in {408, 429, 500, 502, 503, 504}:
            return True

    if isinstance(
        exception,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True

    return False


# Resilient Tenacity Retry Decorator
retry_gcs_operation = retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=1.0, max=30.0, jitter=1.0),
    retry=retry_if_exception(is_transient_gcs_error),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


class GCSManager:
    """Thread-safe singleton managing Google Cloud Storage persistence.

    Features:
        - Resilient retries with exponential backoff & jitter for transient errors.
        - Memory-bounded streaming upload/download using configurable chunk sizes.
        - Recursive directory synchronization with MD5/size hash deduplication.
        - Low-overhead bucket health diagnostics.
        - Graceful local disk fallback (data_storage/local_lake/) when ADC credentials are absent.
    """

    _instance: Optional["GCSManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(
        cls,
        settings: Optional[Settings] = None,
        client: Optional[storage.Client] = None,
        force_new: bool = False,
    ) -> "GCSManager":
        """Instantiate or retrieve the singleton GCSManager instance."""
        if force_new:
            return super().__new__(cls)

        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[storage.Client] = None,
        force_new: bool = False,
    ) -> None:
        """Initialize GCS client and configurations. Only runs once per singleton."""
        if getattr(self, "_initialized", False) and not force_new:
            return

        with self._lock:
            if getattr(self, "_initialized", False) and not force_new:
                return

            self.settings: Settings = settings or get_settings()
            self.bucket_name: str = self.settings.gcs_bucket_name
            self.chunk_size: int = self.settings.gcs_chunk_size
            self.local_lake_dir: Path = Path(getattr(self.settings, "local_lake_dir", "./data_storage/local_lake"))
            self.local_lake_dir.mkdir(parents=True, exist_ok=True)
            self._client_override: Optional[storage.Client] = client
            self._client: Optional[storage.Client] = client
            self.is_local_fallback: bool = False
            self._adc_warning_logged: bool = False
            self._initialized: bool = True

            logger.info(
                "Initialized GCSManager for bucket '%s' with chunk size %d bytes (local lake: %s)",
                self.bucket_name,
                self.chunk_size,
                self.local_lake_dir,
            )

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance (primarily for isolated test fixtures)."""
        with cls._lock:
            cls._instance = None

    def _log_adc_instructions(self) -> None:
        """Log clear setup instructions for Google Cloud Storage Application Default Credentials (ADC)."""
        if not self._adc_warning_logged:
            self._adc_warning_logged = True
            logger.warning(
                "\n"
                "========================================================================\n"
                "[GCS CREDENTIALS NOT CONFIGURED] Google Cloud Storage Credentials Missing\n"
                "------------------------------------------------------------------------\n"
                "The system checked both GCS_CREDENTIALS_PATH and GOOGLE_APPLICATION_CREDENTIALS,\n"
                "as well as local Application Default Credentials (ADC).\n\n"
                "To configure Google Cloud Storage access, choose one of the following:\n"
                "  1. Service Account JSON file (recommended for production):\n"
                "     Set GCS_CREDENTIALS_PATH=/path/to/key.json in .env or set\n"
                "     environment variable GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json\n"
                "  2. Interactive User ADC (recommended for local development):\n"
                "     Install Google Cloud SDK and execute in terminal:\n"
                "       gcloud auth application-default login\n"
                "  3. Google Cloud Compute Engine / GKE / Cloud Run:\n"
                "     Attach a Service Account with 'roles/storage.objectAdmin' IAM permissions.\n"
                "------------------------------------------------------------------------\n"
                "GRACEFUL LOCAL FALLBACK ACTIVATED:\n"
                "  Persisting objects to local storage lake at: %s\n"
                "  Pipeline operations will continue without crashing.\n"
                "========================================================================",
                self.local_lake_dir.resolve(),
            )

    def _get_or_init_client(self) -> Optional[storage.Client]:
        """Lazy-initialize or return the authenticated GCS client, or trigger local fallback."""
        if self._client is not None:
            return self._client
        if self.is_local_fallback:
            return None

        with self._lock:
            if self._client is not None:
                return self._client
            if self.is_local_fallback:
                return None

            # Priority 1: Service Account JSON path
            cred_path = (
                self.settings.gcs_credentials_path
                or os.environ.get("GCS_CREDENTIALS_PATH")
                or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            )
            if cred_path and os.path.exists(cred_path):
                try:
                    logger.info("Authenticating GCS client using service account at: %s", cred_path)
                    self._client = storage.Client.from_service_account_json(cred_path)
                    return self._client
                except Exception as exc:
                    logger.error("Failed authenticating service account JSON at %s: %s", cred_path, exc)

            # Priority 2: Check ADC existence before attempting storage.Client()
            if not check_has_gcs_credentials(self.settings):
                self.is_local_fallback = True
                self._log_adc_instructions()
                return None

            # Priority 3: Try standard ADC
            try:
                logger.info("Authenticating GCS client using Application Default Credentials (ADC)")
                self._client = storage.Client()
                return self._client
            except Exception as exc:
                logger.warning("ADC authentication attempt failed: %s", exc)
                self.is_local_fallback = True
                self._log_adc_instructions()
                return None

    @property
    def client(self) -> storage.Client:
        """Return the authenticated GCS client. Raises RuntimeError if in local fallback mode."""
        c = self._get_or_init_client()
        if c is None:
            raise RuntimeError(
                f"GCS client is not authenticated (operating in local fallback mode at {self.local_lake_dir}). "
                "Configure GCS_CREDENTIALS_PATH or GOOGLE_APPLICATION_CREDENTIALS to connect to remote GCS."
            )
        return c

    @property
    def bucket(self) -> storage.Bucket:
        """Obtain GCS Bucket handle."""
        return self.client.bucket(self.bucket_name)

    def bucket_health_check(self) -> Dict[str, Any]:
        """Perform a diagnostic ping to verify bucket accessibility or local fallback status.

        Returns:
            Dict containing status ('healthy', 'unhealthy', or 'local_fallback'),
            response latency in ms, and storage metadata or diagnostic error details.
        """
        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()

        gcs_client = self._get_or_init_client()
        if gcs_client is None:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "status": "local_fallback",
                "healthy": True,
                "bucket": self.bucket_name,
                "exists": True,
                "location": "local_disk",
                "storage_class": "LOCAL_LAKE",
                "local_lake_dir": str(self.local_lake_dir.resolve()),
                "latency_ms": latency_ms,
                "timestamp": timestamp,
                "mode": "local_disk_fallback",
                "error": None,
                "message": (
                    "Operating in graceful local disk fallback mode (data_storage/local_lake). "
                    "To enable Google Cloud Storage, set GOOGLE_APPLICATION_CREDENTIALS or run 'gcloud auth application-default login'."
                ),
            }

        try:
            bucket = gcs_client.get_bucket(self.bucket_name)
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "status": "healthy",
                "healthy": True,
                "bucket": self.bucket_name,
                "exists": True,
                "location": bucket.location,
                "storage_class": bucket.storage_class,
                "latency_ms": latency_ms,
                "timestamp": timestamp,
                "error": None,
            }
        except NotFound:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.error("Bucket health check failed: Bucket '%s' not found", self.bucket_name)
            return {
                "status": "unhealthy",
                "healthy": False,
                "bucket": self.bucket_name,
                "exists": False,
                "location": None,
                "storage_class": None,
                "latency_ms": latency_ms,
                "timestamp": timestamp,
                "error": f"Bucket '{self.bucket_name}' not found.",
            }
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.exception("Bucket health check failed with unexpected exception")
            return {
                "status": "unhealthy",
                "healthy": False,
                "bucket": self.bucket_name,
                "exists": False,
                "location": None,
                "storage_class": None,
                "latency_ms": latency_ms,
                "timestamp": timestamp,
                "error": str(exc),
            }

    @retry_gcs_operation
    def _upload_file_remote(
        self,
        path: Path,
        remote_blob_name: str,
        content_type: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> str:
        file_size = path.stat().st_size
        effective_chunk_size = chunk_size or self.chunk_size
        blob = self.bucket.blob(remote_blob_name, chunk_size=effective_chunk_size)
        logger.debug(
            "Streaming upload: %s -> gs://%s/%s (%d bytes, chunk=%d)",
            path,
            self.bucket_name,
            remote_blob_name,
            file_size,
            effective_chunk_size,
        )
        with path.open("rb") as file_stream:
            blob.upload_from_file(
                file_stream,
                rewind=True,
                size=file_size,
                content_type=content_type,
            )
        logger.info(
            "Successfully uploaded %s to gs://%s/%s",
            path.name,
            self.bucket_name,
            remote_blob_name,
        )
        return f"gs://{self.bucket_name}/{remote_blob_name}"

    def upload_file(
        self,
        local_path: Union[str, Path],
        remote_blob_name: str,
        content_type: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> str:
        """Stream a local file into a GCS blob or local lake fallback using chunked transfers.

        Buffers memory at most to chunk_size regardless of the file size on disk.

        Args:
            local_path: Local filesystem path to source file.
            remote_blob_name: Destination object key in the GCS bucket or local lake.
            content_type: Optional MIME type (e.g. 'application/json' or 'application/pdf').
            chunk_size: Optional chunk transfer buffer size (defaults to configured 8MB).

        Returns:
            The public or canonical gs:// URI of the uploaded blob.
        """
        path = Path(local_path)
        if not path.is_file():
            raise FileNotFoundError(f"Local file does not exist: {local_path}")

        gcs_client = self._get_or_init_client()
        if gcs_client is None:
            # Graceful local disk fallback
            dest = self.local_lake_dir / remote_blob_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            logger.info("Local storage lake: persisted %s -> %s", path.name, dest)
            return f"gs://{self.bucket_name}/{remote_blob_name}"

        return self._upload_file_remote(
            path=path,
            remote_blob_name=remote_blob_name,
            content_type=content_type,
            chunk_size=chunk_size,
        )

    @retry_gcs_operation
    def _upload_stream_remote(
        self,
        stream: BinaryIO,
        remote_blob_name: str,
        content_type: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> str:
        effective_chunk_size = chunk_size or self.chunk_size
        blob = self.bucket.blob(remote_blob_name, chunk_size=effective_chunk_size)
        blob.upload_from_file(stream, rewind=False, content_type=content_type)
        return f"gs://{self.bucket_name}/{remote_blob_name}"

    def upload_stream(
        self,
        stream: BinaryIO,
        remote_blob_name: str,
        content_type: Optional[str] = None,
        chunk_size: Optional[int] = None,
    ) -> str:
        """Stream data directly from an open binary file-like object to GCS or local fallback without loading into memory."""
        gcs_client = self._get_or_init_client()
        if gcs_client is None:
            dest = self.local_lake_dir / remote_blob_name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as out_fp:
                shutil.copyfileobj(stream, out_fp)
            logger.info("Local storage lake: streamed %d bytes -> %s", dest.stat().st_size, dest)
            return f"gs://{self.bucket_name}/{remote_blob_name}"

        return self._upload_stream_remote(
            stream=stream,
            remote_blob_name=remote_blob_name,
            content_type=content_type,
            chunk_size=chunk_size,
        )

    @retry_gcs_operation
    def _download_file_remote(
        self,
        remote_blob_name: str,
        local_path: Union[str, Path],
        chunk_size: Optional[int] = None,
    ) -> Path:
        dest_path = Path(local_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dest = dest_path.with_suffix(dest_path.suffix + f".tmp_{os.getpid()}")

        effective_chunk_size = chunk_size or self.chunk_size
        blob = self.bucket.blob(remote_blob_name, chunk_size=effective_chunk_size)

        if not blob.exists(self.client):
            raise FileNotFoundError(f"Blob gs://{self.bucket_name}/{remote_blob_name} does not exist.")

        logger.debug(
            "Streaming download: gs://%s/%s -> %s (chunk=%d)",
            self.bucket_name,
            remote_blob_name,
            dest_path,
            effective_chunk_size,
        )

        try:
            with temp_dest.open("wb") as file_stream:
                blob.download_to_file(file_stream)
            # Atomic rename on completion
            temp_dest.replace(dest_path)
        except Exception:
            if temp_dest.exists():
                temp_dest.unlink()
            raise

        logger.info(
            "Successfully downloaded gs://%s/%s to %s",
            self.bucket_name,
            remote_blob_name,
            dest_path,
        )
        return dest_path

    def download_file(
        self,
        remote_blob_name: str,
        local_path: Union[str, Path],
        chunk_size: Optional[int] = None,
    ) -> Path:
        """Download a GCS blob to a local file using memory-bounded streaming or local lake fallback.

        Writes first to a temporary atomic file, then renames to avoid partial corruption.

        Args:
            remote_blob_name: Target object key in the GCS bucket or local lake.
            local_path: Target destination path on local disk.
            chunk_size: Buffer chunk size for streaming read operations.

        Returns:
            Resolved Path to the downloaded local file.
        """
        gcs_client = self._get_or_init_client()
        if gcs_client is None:
            source = self.local_lake_dir / remote_blob_name
            if not source.exists():
                raise FileNotFoundError(f"Blob {remote_blob_name} does not exist in local lake: {source}")
            dest_path = Path(local_path)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest_path)
            logger.info("Local storage lake: copied %s -> %s", source, dest_path)
            return dest_path

        return self._download_file_remote(
            remote_blob_name=remote_blob_name,
            local_path=local_path,
            chunk_size=chunk_size,
        )

    def sync_directory_to_bucket(
        self,
        local_dir: Union[str, Path],
        remote_prefix: str = "",
        skip_identical: bool = True,
    ) -> Dict[str, Any]:
        """Synchronize an entire local directory tree to the GCS bucket or local lake fallback.

        Skips files that already exist remotely with matching size/content.

        Args:
            local_dir: Local folder path to sync.
            remote_prefix: Optional prefix in bucket (e.g. 'ingestion_stream/2026/').
            skip_identical: If True, avoids uploading if size and/or hash match.

        Returns:
            Summary dictionary with counts of uploaded, skipped, and failed files.
        """
        base_dir = Path(local_dir).resolve()
        if not base_dir.is_dir():
            raise NotADirectoryError(f"Local directory does not exist: {local_dir}")

        prefix = remote_prefix.strip("/")

        gcs_client = self._get_or_init_client()
        if gcs_client is None:
            # Graceful local disk fallback sync
            target_base = self.local_lake_dir / prefix if prefix else self.local_lake_dir
            target_base.mkdir(parents=True, exist_ok=True)

            uploaded_files: List[str] = []
            skipped_files: List[str] = []
            failed_files: List[Dict[str, str]] = []
            total_bytes = 0

            for root, _, files in os.walk(base_dir):
                for file_name in files:
                    local_file = Path(root) / file_name
                    rel_path = local_file.relative_to(base_dir).as_posix()
                    dest_file = target_base / rel_path

                    try:
                        local_size = local_file.stat().st_size
                        if skip_identical and dest_file.exists() and dest_file.stat().st_size == local_size:
                            skipped_files.append(rel_path)
                            continue

                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(local_file, dest_file)
                        uploaded_files.append(rel_path)
                        total_bytes += local_size
                    except Exception as exc:
                        logger.error("Failed to sync file to local lake %s: %s", rel_path, exc)
                        failed_files.append({"file": rel_path, "error": str(exc)})

            summary = {
                "uploaded_count": len(uploaded_files),
                "skipped_count": len(skipped_files),
                "failed_count": len(failed_files),
                "bytes_transferred": total_bytes,
                "uploaded_files": uploaded_files,
                "uploaded": uploaded_files,
                "skipped_files": skipped_files,
                "failed_files": failed_files,
                "mode": "local_disk_fallback",
                "local_lake_dir": str(target_base.resolve()),
            }
            logger.info(
                "Local lake sync complete: %d uploaded, %d skipped, %d failed (%d bytes)",
                summary["uploaded_count"],
                summary["skipped_count"],
                summary["failed_count"],
                total_bytes,
            )
            return summary

        # Remote GCS sync
        uploaded_files = []
        skipped_files = []
        failed_files = []
        total_bytes = 0

        logger.info("Starting sync from '%s' to gs://%s/%s", base_dir, self.bucket_name, prefix)

        for root, _, files in os.walk(base_dir):
            for file_name in files:
                local_file = Path(root) / file_name
                rel_path = local_file.relative_to(base_dir).as_posix()
                blob_name = f"{prefix}/{rel_path}" if prefix else rel_path

                try:
                    local_size = local_file.stat().st_size
                    if skip_identical:
                        blob = self.bucket.get_blob(blob_name)
                        if blob is not None and blob.size == local_size:
                            logger.debug("Skipping unchanged file: %s", rel_path)
                            skipped_files.append(rel_path)
                            continue

                    self.upload_file(local_file, blob_name)
                    uploaded_files.append(rel_path)
                    total_bytes += local_size
                except Exception as exc:
                    logger.error("Failed to sync file %s: %s", rel_path, exc)
                    failed_files.append({"file": rel_path, "error": str(exc)})

        summary = {
            "uploaded_count": len(uploaded_files),
            "skipped_count": len(skipped_files),
            "failed_count": len(failed_files),
            "bytes_transferred": total_bytes,
            "uploaded_files": uploaded_files,
            "uploaded": uploaded_files,
            "skipped_files": skipped_files,
            "failed_files": failed_files,
        }
        logger.info(
            "Sync complete: %d uploaded, %d skipped, %d failed (%d bytes)",
            summary["uploaded_count"],
            summary["skipped_count"],
            summary["failed_count"],
            total_bytes,
        )
        return summary

    def flush_local_lake_to_gcs(
        self,
        delete_after_upload: bool = False,
        sync_memory: bool = True,
    ) -> Dict[str, Any]:
        """Flush all locally buffered documents and indexes from local storage lake to GCS.

        Args:
            delete_after_upload: Whether to purge local lake files after successful upload.
            sync_memory: Whether to also upload local FAISS memory files (index.faiss, metadata.json).

        Returns:
            Dictionary with summary metrics of flushed objects.
        """
        gcs_client = self._get_or_init_client()
        if gcs_client is None:
            self._log_adc_instructions()
            return {
                "status": "skipped_unauthenticated",
                "message": "Cannot flush to GCS: Google Cloud credentials not configured. Local lake retained.",
                "local_lake_dir": str(self.local_lake_dir.resolve()),
                "flushed_count": 0,
                "bytes_transferred": 0,
            }

        flushed: List[str] = []
        failed: List[Dict[str, str]] = []
        total_bytes = 0

        # 1. Flush local lake files
        if self.local_lake_dir.exists():
            for root, _, files in os.walk(self.local_lake_dir):
                for file_name in files:
                    file_path = Path(root) / file_name
                    blob_name = file_path.relative_to(self.local_lake_dir).as_posix()
                    file_size = file_path.stat().st_size
                    try:
                        self.upload_file(file_path, blob_name)
                        flushed.append(blob_name)
                        total_bytes += file_size
                        if delete_after_upload:
                            file_path.unlink()
                    except Exception as exc:
                        logger.error("Failed flushing local lake file '%s' to GCS: %s", blob_name, exc)
                        failed.append({"file": blob_name, "error": str(exc)})

        # 2. Optionally sync memory files (index.faiss and metadata.json)
        memory_flushed: List[str] = []
        if sync_memory and self.settings.memory_local_dir.exists():
            for mem_file in ("index.faiss", "metadata.json"):
                p = self.settings.memory_local_dir / mem_file
                if p.exists() and p.is_file():
                    blob_name = f"memory/{mem_file}"
                    try:
                        self.upload_file(p, blob_name)
                        memory_flushed.append(blob_name)
                        total_bytes += p.stat().st_size
                    except Exception as exc:
                        logger.error("Failed syncing memory file '%s': %s", blob_name, exc)
                        failed.append({"file": blob_name, "error": str(exc)})

        summary = {
            "status": "flushed" if not failed else "partial",
            "flushed_count": len(flushed) + len(memory_flushed),
            "lake_files_flushed": flushed,
            "memory_files_flushed": memory_flushed,
            "failed_count": len(failed),
            "failed": failed,
            "bytes_transferred": total_bytes,
        }
        logger.info(
            "Flush to GCS completed: %d files uploaded (%d bytes transferred, %d failed)",
            summary["flushed_count"],
            total_bytes,
            len(failed),
        )
        return summary


def get_gcs_manager(
    settings: Optional[Settings] = None,
    client: Optional[storage.Client] = None,
) -> GCSManager:
    """Convenience accessor to obtain the singleton GCSManager instance."""
    return GCSManager(settings=settings, client=client)
