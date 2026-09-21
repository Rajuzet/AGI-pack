"""CLI runner and orchestrator for data ingestion, vector memory, and GCS persistence."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
from typing import Optional

from config import get_settings
from ingestion.stream_sources import StreamIngestionClient
from memory.vector_store import VectorStore
from storage.gcs_manager import get_gcs_manager
from storage.mysql_manager import get_mysql_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agi_data_engine")


async def run_ingestion(
    source: str = "all",
    limit: int = 10,
    stage: bool = True,
    sync_gcs: bool = False,
    dry_run: bool = False,
) -> None:
    """Execute ingestion and persistence cycle."""
    settings = get_settings()
    client = StreamIngestionClient(settings=settings)
    gcs_mgr = get_gcs_manager(settings=settings)

    logger.info("Initiating ingestion cycle (source=%s, limit=%d, dry_run=%s)", source, limit, dry_run)

    if source == "arxiv":
        records = await client.fetch_arxiv(max_results=limit)
    elif source == "rss":
        records = await client.fetch_all_rss_feeds(max_entries_per_feed=limit)
    else:
        records = await client.fetch_all_sources(arxiv_limit=limit, rss_limit=limit)

    logger.info("Ingestion completed: %d total records harvested.", len(records))

    if not records:
        logger.warning("No records harvested in this cycle.")
        return

    # Print sample record
    sample = records[0]
    logger.info("Sample record extracted:\n%s", json.dumps(sample.model_dump(), indent=2)[:500] + "...\n")

    if stage:
        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        staging_file = settings.staging_dir / f"stream_{source}_{now_str}.jsonl"
        saved_path = client.save_records_to_jsonl(records, staging_file)
        logger.info("Staged records to local file: %s (%d bytes)", saved_path, saved_path.stat().st_size)

        # Audit log harvested papers to MySQL (non-blocking)
        mysql_mgr = get_mysql_manager(settings=settings)
        recorded_count = 0
        for rec in records:
            if mysql_mgr.record_paper(rec.model_dump()):
                recorded_count += 1
        if recorded_count > 0:
            logger.info("Audited %d/%d harvested records into MySQL relational storage.", recorded_count, len(records))

        if sync_gcs:
            if dry_run:
                logger.info("[DRY RUN] Would upload staged file '%s' to GCS bucket '%s'", saved_path, settings.gcs_bucket_name)
            else:
                remote_blob = f"ingestion_stream/{now_str[:8]}/{saved_path.name}"
                logger.info("Streaming staged file to GCS: gs://%s/%s", settings.gcs_bucket_name, remote_blob)
                try:
                    uri = gcs_mgr.upload_file(saved_path, remote_blob, content_type="application/x-ndjson")
                    logger.info("Upload confirmed: %s", uri)
                except Exception as exc:
                    logger.error("GCS upload failed: %s", exc)


def run_health_check() -> None:
    """Run unified infrastructure diagnostics for both GCS and MySQL with latency metrics."""
    settings = get_settings()
    gcs_mgr = get_gcs_manager(settings=settings)
    mysql_mgr = get_mysql_manager(settings=settings)

    logger.info(
        "Executing unified infrastructure health check (GCS: '%s', MySQL: '%s:%s')...",
        settings.gcs_bucket_name,
        settings.mysql_host,
        settings.mysql_port,
    )

    gcs_diag = gcs_mgr.bucket_health_check()
    mysql_diag = mysql_mgr.health_check()

    # Determine aggregated system status
    gcs_ok = gcs_diag.get("healthy", False)
    mysql_ok = mysql_diag.get("healthy", False)
    if gcs_ok and mysql_ok:
        overall_status = "healthy"
    elif gcs_ok or mysql_ok:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall_status,
        "cloud_storage_gcs": gcs_diag,
        "relational_audit_mysql": mysql_diag,
    }
    print(json.dumps(report, indent=2))


def index_staged_records(staging_dir: Optional[Path] = None) -> None:
    """Read staged JSONL records, chunk them, embed, and store in FAISS vector memory."""
    settings = get_settings()
    sdir = staging_dir or settings.staging_dir
    vector_store = VectorStore(settings=settings)

    # Load existing local index if available
    vector_store.load_local()

    jsonl_files = list(sdir.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning("No staged .jsonl files found in %s to index.", sdir)
        return

    total_records = []
    for jf in jsonl_files:
        logger.info("Loading documents from staged file: %s", jf.name)
        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        total_records.append(json.loads(line_str))
                    except Exception as e:
                        logger.warning("Failed parsing line: %s", e)

    logger.info("Indexing %d documents into FAISS vector store...", len(total_records))
    added_ids = vector_store.add_documents(total_records, chunk_documents=True)
    saved_path = vector_store.save_local()
    logger.info(
        "Indexing complete: added %d new vector passages. Total vectors in index: %d. Saved to: %s",
        len(added_ids),
        vector_store.index.ntotal,
        saved_path,
    )


def query_memory(query_text: str, top_k: int = 5) -> None:
    """Query the vector store and display top-k nearest semantic matches."""
    settings = get_settings()
    vector_store = VectorStore(settings=settings)
    loaded = vector_store.load_local()
    if not loaded and vector_store.index.ntotal == 0:
        logger.warning("Local vector memory is empty. Run with --index-staged first.")
        return

    logger.info("Searching vector memory for: '%s' (top_k=%d)...", query_text, top_k)
    results = vector_store.similarity_search(query_text, top_k=top_k)

    if not results:
        print("No matching records found.")
        return

    print(f"\n--- Top {len(results)} Matches for Query: '{query_text}' ---")
    for i, res in enumerate(results, 1):
        print(f"\n[{i}] Score: {res['score']:.4f} | ID: {res['id']}")
        print(f"Title: {res.get('title') or 'N/A'}")
        print(f"Source: {res.get('source')}")
        text_preview = res['text'][:250].replace('\n', ' ') + ("..." if len(res['text']) > 250 else "")
        print(f"Passage: {text_preview}")
        if res.get("metadata", {}).get("pdf_url"):
            print(f"PDF: {res['metadata']['pdf_url']}")
        if res.get("metadata", {}).get("link"):
            print(f"Link: {res['metadata']['link']}")


def sync_memory_gcs(to_gcs: bool = True, dry_run: bool = False) -> None:
    """Synchronize vector memory index and metadata to/from Google Cloud Storage."""
    settings = get_settings()
    gcs_mgr = get_gcs_manager(settings=settings)
    vector_store = VectorStore(settings=settings)

    if to_gcs:
        vector_store.load_local()
        if dry_run:
            logger.info("[DRY RUN] Would flush %d vectors from %s to gs://%s/%s",
                        vector_store.index.ntotal, settings.memory_local_dir,
                        settings.gcs_bucket_name, settings.memory_remote_prefix)
        else:
            summary = vector_store.sync_to_gcs(gcs_mgr)
            print(json.dumps(summary, indent=2))
    else:
        if dry_run:
            logger.info("[DRY RUN] Would pull vector memory from gs://%s/%s to %s",
                        settings.gcs_bucket_name, settings.memory_remote_prefix,
                        settings.memory_local_dir)
        else:
            loaded = vector_store.sync_from_gcs(gcs_mgr)
            logger.info("Remote sync completed. Loaded existing index: %s. Total vectors: %d",
                        loaded, vector_store.index.ntotal)


def run_agent(
    query_text: str,
    as_json: bool = False,
    use_serving: bool = False,
    serving_url: Optional[str] = None,
) -> None:
    """Execute autonomous ReAct reasoning agent with error self-correction and relational audit."""
    from agent.orchestrator import ReActOrchestrator

    logger.info(
        "Initializing Autonomous ReAct Orchestrator for query: '%s' (use_serving=%s)...",
        query_text,
        use_serving,
    )
    settings = get_settings()
    mysql_mgr = get_mysql_manager(settings=settings)

    serving_endpoint = None
    if use_serving or serving_url:
        host = "127.0.0.1" if settings.serving_host in ("0.0.0.0", "::") else settings.serving_host
        serving_endpoint = serving_url or f"http://{host}:{settings.serving_port}/v1/chat/completions"
        logger.info("Connecting ReAct orchestrator to local serving endpoint: %s", serving_endpoint)

    orchestrator = ReActOrchestrator(
        audit_manager=mysql_mgr,
        serving_endpoint=serving_endpoint,
    )
    result = orchestrator.run(query_text)

    if as_json:
        print(result.to_json())
    else:
        print("\n" + result.to_markdown() + "\n")


def run_serving_server(host: Optional[str] = None, port: Optional[int] = None) -> None:
    """Launch production FastAPI model serving microservice."""
    from serving.app import start_server
    start_server(host=host, port=port)


def run_ingestion_daemon(once: bool = False) -> None:
    """Run continuous data harvesting and vectorization daemon."""
    from daemon.continuous_ingest import ContinuousIngestionDaemon, run_daemon
    if once:
        daemon = ContinuousIngestionDaemon()
        summary = asyncio.run(daemon.run_once())
        print("\n--- Ingestion Daemon Single Cycle Summary ---")
        print(json.dumps(summary, indent=2))
    else:
        run_daemon()


def run_training_pipeline(
    model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    epochs: int = 1,
    dry_run: bool = False,
    upload_gcs: bool = True,
) -> None:
    """Execute free-tier cloud adaptation QLoRA training pipeline."""
    from training.kaggle_lora_train import LoRATrainingConfig, run_lora_training

    logger.info("Configuring QLoRA training pipeline for model '%s'...", model_id)
    config = LoRATrainingConfig(
        model_id=model_id,
        num_train_epochs=epochs,
        dry_run=dry_run,
        upload_gcs=upload_gcs,
    )
    summary = run_lora_training(config=config)

    # Relational audit logging of training run to MySQL
    settings = get_settings()
    mysql_mgr = get_mysql_manager(settings=settings)
    mysql_mgr.record_training_run(summary)

    print("\n--- Training Pipeline Summary ---")
    print(json.dumps(summary, indent=2))


def main() -> None:
    """CLI argument parsing and execution router."""
    parser = argparse.ArgumentParser(
        description="AGI Autonomous Agent Data Ingestion, Vector Memory, ReAct Core & Cloud Training"
    )
    # Agent Reasoning args
    parser.add_argument(
        "--agent-run",
        type=str,
        default=None,
        help="Execute autonomous ReAct reasoning engine on user objective",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON response instead of synthesized Markdown",
    )
    parser.add_argument(
        "--use-serving",
        action="store_true",
        help="Route agent reasoning requests through local FastAPI model serving microservice",
    )
    parser.add_argument(
        "--serving-url",
        type=str,
        default=None,
        help="Override model serving endpoint URL (default: http://localhost:8000/v1/chat/completions)",
    )

    # Serving & Daemon Service args
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the production FastAPI model serving microservice",
    )
    parser.add_argument(
        "--serve-host",
        type=str,
        default=None,
        help="Host address to bind serving microservice (default: from config/0.0.0.0)",
    )
    parser.add_argument(
        "--serve-port",
        type=int,
        default=None,
        help="Port number to bind serving microservice (default: from config/8000)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run continuous background data harvesting and vectorization daemon",
    )
    parser.add_argument(
        "--daemon-once",
        action="store_true",
        help="Execute a single continuous ingestion daemon cycle and exit",
    )

    # Cloud Training args
    parser.add_argument(
        "--train-lora",
        action="store_true",
        help="Run parameter-efficient QLoRA fine-tuning pipeline",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model repository ID for training (default: Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of fine-tuning epochs (default: 1)",
    )

    # Ingestion args
    parser.add_argument(
        "--source",
        choices=["arxiv", "rss", "all"],
        default=None,
        help="Data source to ingest (arxiv, rss, all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum records to fetch per source (default: 5)",
    )
    parser.add_argument(
        "--no-stage",
        action="store_true",
        help="Disable staging records to local JSONL files",
    )
    parser.add_argument(
        "--sync-gcs",
        action="store_true",
        help="Stream staged JSONL file to target GCS bucket",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate operations without remote state modifications",
    )
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Perform unified diagnostic health check on GCS bucket and MySQL database and exit",
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize relational audit tables in MySQL using storage/schema.sql",
    )

    # Memory engine args
    parser.add_argument(
        "--index-staged",
        action="store_true",
        help="Read staged JSONL records, chunk text, and index into FAISS vector store",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run semantic similarity search against the vector memory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest neighbor matches to retrieve for --query (default: 5)",
    )
    parser.add_argument(
        "--sync-memory-to-gcs",
        action="store_true",
        help="Push local vector memory (index.faiss, metadata.json) directly to GCS bucket",
    )
    parser.add_argument(
        "--sync-memory-from-gcs",
        action="store_true",
        help="Pull vector memory from GCS bucket (handles cold start gracefully)",
    )
    parser.add_argument(
        "--flush-lake",
        action="store_true",
        help="Flush all documents from local storage lake and memory to live GCS bucket",
    )
    parser.add_argument(
        "--delete-flushed",
        action="store_true",
        help="Purge local files after confirmed flush to GCS",
    )

    args = parser.parse_args()

    # Priority 0: Production Services & Background Daemons
    if args.serve:
        run_serving_server(host=args.serve_host, port=args.serve_port)
        return

    if args.daemon:
        run_ingestion_daemon(once=False)
        return

    if args.daemon_once:
        run_ingestion_daemon(once=True)
        return

    # Priority 1: Agent Reasoning
    if args.agent_run:
        run_agent(
            args.agent_run,
            as_json=args.json,
            use_serving=args.use_serving,
            serving_url=args.serving_url,
        )
        return

    # Priority 2: Training Pipeline
    if args.train_lora:
        run_training_pipeline(
            model_id=args.model_id,
            epochs=args.epochs,
            dry_run=args.dry_run,
            upload_gcs=not args.dry_run,
        )
        return

    # Priority 3: Diagnostics & Maintenance
    if args.health_check:
        run_health_check()
        return

    if args.init_db:
        settings = get_settings()
        mysql_mgr = get_mysql_manager(settings=settings)
        success = mysql_mgr.init_schema()
        if success:
            logger.info("Successfully provisioned relational audit schema in MySQL.")
        else:
            logger.warning("Could not provision MySQL schema (ensure MySQL is running or check credentials).")
        return

    if args.index_staged:
        index_staged_records()
        return

    if args.query:
        query_memory(args.query, top_k=args.top_k)
        return

    if args.sync_memory_to_gcs:
        sync_memory_gcs(to_gcs=True, dry_run=args.dry_run)
        return

    if args.sync_memory_from_gcs:
        sync_memory_gcs(to_gcs=False, dry_run=args.dry_run)
        return

    if args.flush_lake:
        settings = get_settings()
        gcs_mgr = get_gcs_manager(settings=settings)
        flush_summary = gcs_mgr.flush_local_lake_to_gcs(
            delete_after_upload=args.delete_flushed,
            sync_memory=True,
        )
        print(json.dumps(flush_summary, indent=2))
        return

    if args.source:
        asyncio.run(
            run_ingestion(
                source=args.source,
                limit=args.limit,
                stage=not args.no_stage,
                sync_gcs=args.sync_gcs,
                dry_run=args.dry_run,
            )
        )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
