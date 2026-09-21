"""Continuous background ingestion and vectorization daemon.

Autonomously harvests scientific papers (ArXiv) and technical RSS feeds,
deduplicates against MySQL relational memory, chunks and indexes into FAISS,
and synchronizes both staged data lakes and vector indices to Google Cloud Storage.
"""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import signal
import sys
import time
from typing import Any, Dict, List, Optional, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Settings, get_settings
from ingestion.stream_sources import StandardizedRecord, StreamIngestionClient
from memory.vector_store import VectorStore
from monitoring.metrics import record_paper_ingested, update_system_gauges
from prometheus_client import start_http_server
from storage.gcs_manager import GCSManager, get_gcs_manager
from storage.mysql_manager import MySQLAuditManager, get_mysql_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [agi_daemon] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agi_daemon")


class ContinuousIngestionDaemon:
    """Production background worker for persistent data harvesting and vector memory sync."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings: Settings = settings or get_settings()
        self.poll_interval: int = self.settings.daemon_poll_interval
        self.max_per_source: int = self.settings.daemon_max_records_per_poll

        self.ingestion_client = StreamIngestionClient(settings=self.settings)
        self.vector_store = VectorStore(settings=self.settings)
        self.gcs_mgr: GCSManager = get_gcs_manager(settings=self.settings)
        self.mysql_mgr: MySQLAuditManager = get_mysql_manager(settings=self.settings)

        self._stop_event = asyncio.Event()
        self.is_running: bool = False
        self.cycle_count: int = 0
        self.arxiv_offset: int = 15
        self.stats: Dict[str, int] = {
            "total_harvested": 0,
            "total_deduped": 0,
            "total_indexed": 0,
            "total_synced": 0,
        }

        # Attempt to warm-load existing FAISS vector store
        try:
            self.vector_store.load_local()
        except Exception as exc:
            logger.warning("Could not load initial local FAISS index: %s", exc)

    def _compute_content_hash(self, text: str) -> str:
        """Compute SHA256 fingerprint of normalized document text."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    async def run_once(self) -> Dict[str, Any]:
        """Execute a single atomic harvesting, deduplication, indexing, and sync cycle."""
        self.cycle_count += 1
        cycle_start = time.time()
        now_utc = datetime.now(timezone.utc)
        logger.info("=== Starting Ingestion Daemon Cycle #%d at %s ===", self.cycle_count, now_utc.isoformat())

        # Step 1: Harvest papers and technical RSS feeds
        logger.info("Harvesting from ArXiv (%s, offset=%d) and RSS feeds (limit=%d)...",
                    self.settings.arxiv_categories, self.arxiv_offset, self.max_per_source)
        try:
            records: List[StandardizedRecord] = await self.ingestion_client.fetch_all_sources(
                arxiv_limit=self.max_per_source,
                rss_limit=self.max_per_source,
                arxiv_start=self.arxiv_offset,
            )
            self.arxiv_offset += self.max_per_source
        except Exception as exc:
            logger.error("Harvesting failed in cycle #%d: %s", self.cycle_count, exc)
            return {
                "cycle": self.cycle_count,
                "status": "failed",
                "error": str(exc),
                "duration_seconds": round(time.time() - cycle_start, 2),
            }

        logger.info("Harvested %d candidate records.", len(records))
        self.stats["total_harvested"] += len(records)

        if not records:
            logger.info("No records harvested in this cycle. Concluding cycle.")
            return {
                "cycle": self.cycle_count,
                "status": "empty",
                "harvested": 0,
                "new_indexed": 0,
                "duration_seconds": round(time.time() - cycle_start, 2),
            }

        # Step 2: Deduplication against MySQL relational memory and VectorStore catalog
        candidate_ids = [r.id for r in records]
        existing_ids: Set[str] = self.mysql_mgr.get_existing_paper_ids(candidate_ids)
        if hasattr(self.vector_store, "doc_ids") and self.vector_store.doc_ids:
            existing_ids = existing_ids.union(
                {cid for cid in candidate_ids if cid in self.vector_store.doc_ids or f"{cid}_chunk_0" in self.vector_store.doc_ids}
            )
        logger.info("Deduplication check: %d/%d records already exist in database/memory.",
                    len(existing_ids), len(records))

        new_records: List[StandardizedRecord] = [
            r for r in records if r.id not in existing_ids
        ]
        self.stats["total_deduped"] += (len(records) - len(new_records))

        if not new_records:
            logger.info("All harvested items are already indexed. No new documents to vectorize.")
            return {
                "cycle": self.cycle_count,
                "status": "deduped",
                "harvested": len(records),
                "new_indexed": 0,
                "existing_skipped": len(existing_ids),
                "duration_seconds": round(time.time() - cycle_start, 2),
            }

        logger.info("Proceeding with %d newly discovered research documents.", len(new_records))

        # Step 3: Stage records locally to JSONL and audit in MySQL
        date_str = now_utc.strftime("%Y%m%d")
        time_str = now_utc.strftime("%H%M%S")
        staging_filename = f"daemon_harvest_{date_str}_{time_str}.jsonl"
        staging_file = self.settings.staging_dir / staging_filename

        saved_path = self.ingestion_client.save_records_to_jsonl(new_records, staging_file)
        logger.info("Staged %d records to: %s (%d bytes)",
                    len(new_records), saved_path, saved_path.stat().st_size)

        # Audit into MySQL research_papers table
        audited_count = 0
        for r in new_records:
            record_paper_ingested(domain=r.source or "unknown")
            if self.mysql_mgr.record_paper(r.model_dump()):
                audited_count += 1
        logger.info("Audited %d/%d new records in MySQL relational database.", audited_count, len(new_records))

        # Step 4: Stream staged JSONL to GCS Data Lake
        remote_blob = f"ingestion_stream/{date_str}/{saved_path.name}"
        logger.info("Streaming staged batch to GCS data lake: gs://%s/%s",
                    self.settings.gcs_bucket_name, remote_blob)
        gcs_uri = None
        try:
            gcs_uri = self.gcs_mgr.upload_file(saved_path, remote_blob, content_type="application/x-ndjson")
            logger.info("GCS upload confirmed: %s", gcs_uri)
        except Exception as exc:
            logger.warning("GCS batch streaming failed (offline or credentials missing): %s", exc)

        # Step 5: Semantic Chunking and FAISS Vectorization
        raw_dicts = [r.model_dump() for r in new_records]
        logger.info("Chunking and vectorizing %d new documents into FAISS vector memory...", len(raw_dicts))
        added_vector_ids = self.vector_store.add_documents(raw_dicts, chunk_documents=True)
        self.stats["total_indexed"] += len(added_vector_ids)

        saved_vector_path = self.vector_store.save_local()
        total_vectors = self.vector_store.index.ntotal if self.vector_store.index else 0
        update_system_gauges(
            active_vectors=total_vectors,
            buffered_count=self.mysql_mgr.buffered_count,
        )
        logger.info("FAISS vector store updated: added %d vectors. Total in memory: %d. Saved to: %s",
                    len(added_vector_ids), total_vectors, saved_vector_path)

        # Step 6: Synchronize Vector Memory Index to GCS
        logger.info("Synchronizing FAISS index to GCS prefix: gs://%s/%s...",
                    self.settings.gcs_bucket_name, self.settings.memory_remote_prefix)
        sync_result = {}
        try:
            sync_result = self.vector_store.sync_to_gcs(self.gcs_mgr)
            self.stats["total_synced"] += 1
            logger.info("GCS Vector memory sync confirmed: %s", sync_result.get("status"))
        except Exception as exc:
            logger.warning("Vector memory GCS sync skipped/failed: %s", exc)

        duration = round(time.time() - cycle_start, 2)
        logger.info("=== Concluded Cycle #%d in %.2fs (New: %d, Vectors Added: %d) ===",
                    self.cycle_count, duration, len(new_records), len(added_vector_ids))

        return {
            "cycle": self.cycle_count,
            "status": "success",
            "harvested": len(records),
            "new_indexed": len(new_records),
            "vectors_added": len(added_vector_ids),
            "total_vectors_in_store": total_vectors,
            "staged_file": str(saved_path),
            "gcs_lake_uri": gcs_uri,
            "duration_seconds": duration,
            "timestamp": now_utc.isoformat(),
        }

    def _setup_signal_handlers(self) -> None:
        """Register POSIX and Windows signal handlers for graceful shutdown."""
        def handle_signal(sig: int, frame: Any) -> None:
            sig_name = signal.Signals(sig).name if hasattr(signal, "Signals") else str(sig)
            logger.info("Received termination signal (%s). Gracefully stopping daemon...", sig_name)
            self.stop()

        try:
            signal.signal(signal.SIGINT, handle_signal)
            signal.signal(signal.SIGTERM, handle_signal)
        except Exception as exc:
            logger.debug("Signal handler registration note: %s", exc)

    def stop(self) -> None:
        """Signal the daemon loop to stop after completing the current cycle."""
        self.is_running = False
        self._stop_event.set()
        logger.info("Continuous Ingestion Daemon signaled to stop.")

    async def start(self, metrics_port: int = 9090, max_cycles: int = 0) -> None:
        """Run the persistent continuous harvesting and vectorization loop."""
        self._setup_signal_handlers()
        try:
            start_http_server(metrics_port)
            logger.info("Prometheus metrics server started on port %d", metrics_port)
        except Exception as exc:
            logger.warning("Prometheus metrics server could not be started on port %d: %s", metrics_port, exc)

        self.is_running = True
        self._stop_event.clear()
        logger.info("Continuous Ingestion Daemon started. Polling every %d seconds.", self.poll_interval)

        completed_cycles = 0
        while self.is_running and not self._stop_event.is_set():
            try:
                await self.run_once()
                completed_cycles += 1
                if max_cycles > 0 and completed_cycles >= max_cycles:
                    logger.info("Completed %d supervised cycle(s). Stopping daemon.", completed_cycles)
                    break
            except Exception as exc:
                logger.error("Unhandled exception in daemon loop: %s", exc, exc_info=True)

            logger.info("Daemon sleeping for %d seconds until next harvest cycle...", self.poll_interval)
            try:
                # Wait for next poll or interrupt
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.poll_interval)
            except asyncio.TimeoutError:
                # Normal timer expiry; proceed to next loop iteration
                pass

        logger.info("Continuous Ingestion Daemon exited cleanly.")


def run_daemon(
    poll_interval: Optional[int] = None,
    daemon_once: bool = False,
    metrics_port: int = 9090,
    cycles: int = 0,
) -> None:
    """CLI entrypoint to run the ingestion daemon standing service."""
    import argparse
    parser = argparse.ArgumentParser(description="Continuous Ingestion Daemon")
    parser.add_argument("--daemon-once", action="store_true", help="Execute a single harvesting cycle and exit")
    parser.add_argument("--poll-interval", type=int, default=None, help="Poll interval in seconds")
    parser.add_argument("--metrics-port", type=int, default=9090, help="Prometheus metrics server port (default: 9090)")
    parser.add_argument("--cycles", type=int, default=0, help="Number of cycles to run before exiting (0 = infinite)")
    parser.add_argument("--max-records", type=int, default=None, help="Max records per source per cycle")
    args, _ = parser.parse_known_args()

    settings = get_settings()
    target_interval = args.poll_interval if args.poll_interval is not None else poll_interval
    if target_interval is not None:
        settings.daemon_poll_interval = int(target_interval)
    if args.max_records is not None:
        settings.daemon_max_records_per_poll = int(args.max_records)

    daemon = ContinuousIngestionDaemon(settings=settings)
    target_cycles = args.cycles if args.cycles else cycles
    try:
        if args.daemon_once or daemon_once:
            logger.info("Executing Continuous Ingestion Daemon in single-cycle mode (--daemon-once)...")
            result = asyncio.run(daemon.run_once())
            logger.info("Daemon single-cycle execution finished: %s", result)
        else:
            asyncio.run(daemon.start(metrics_port=args.metrics_port or metrics_port, max_cycles=target_cycles))
    except (KeyboardInterrupt, SystemExit):
        logger.info("Daemon interrupted. Exiting.")


if __name__ == "__main__":
    run_daemon()
