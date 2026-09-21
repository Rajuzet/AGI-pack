"""Production Load Testing & Performance Benchmarking Harness.

Validates end-to-end performance of:
1. Inference Engine (POST /v1/chat/completions):
   - Measures Time to First Token (TTFT), Tokens Per Second (TPS), P50/P95/P99 latency,
     and peak VRAM/system RAM overhead across 5, 10, and 20 concurrent async workers.
2. Ingestion & Vector Daemon (daemon/continuous_ingest.py):
   - Measures raw harvesting throughput (records/sec), embedding computation rate
     (embeddings/sec), and FAISS vector index insertion latency.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import psutil
import torch

from config import Settings, get_settings
from daemon.continuous_ingest import ContinuousIngestionDaemon
from ingestion.stream_sources import StandardizedRecord
from memory.vector_store import VectorStore
from serving.app import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [benchmark] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agi_benchmark")


# ---------------------------------------------------------------------------
# Metric Collectors & Utility Helpers
# ---------------------------------------------------------------------------

def get_system_ram_mb() -> float:
    """Return total process RSS memory overhead in megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_vram_mb() -> float:
    """Return PyTorch CUDA VRAM allocated in megabytes (0 if CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate min, mean, P50, P95, P99, and max from a list of numerical values."""
    if not values:
        return {"min": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def p(pct: float) -> float:
        idx = int(round(pct * (n - 1)))
        return sorted_vals[min(idx, n - 1)]

    return {
        "min": round(min(sorted_vals), 4),
        "mean": round(statistics.mean(sorted_vals), 4),
        "p50": round(statistics.median(sorted_vals), 4),
        "p95": round(p(0.95), 4),
        "p99": round(p(0.99), 4),
        "max": round(max(sorted_vals), 4),
    }


# ---------------------------------------------------------------------------
# 1. Inference Engine Concurrency Load Test
# ---------------------------------------------------------------------------

async def _send_single_chat_request(
    client: httpx.AsyncClient,
    base_url: str,
    prompt: str,
    request_id: int,
    stream: bool = True,
) -> Dict[str, Any]:
    """Send a single chat completion request and measure TTFT, TPS, and total latency."""
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "system", "content": "You are an autonomous AI research scientist."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 128,
        "stream": stream,
    }

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    first_token_recorded = False
    tokens_received = 0
    full_content = []

    try:
        if stream:
            async with client.stream("POST", url, json=payload, timeout=60.0) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise RuntimeError(f"HTTP {response.status_code}: {error_body.decode('utf-8')}")

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk_str = line[6:].strip()
                        try:
                            chunk_json = json.loads(chunk_str)
                            choices = chunk_json.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                token_text = delta.get("content", "")
                                if token_text:
                                    if not first_token_recorded:
                                        ttft = time.perf_counter() - t0
                                        first_token_recorded = True
                                    tokens_received += len(token_text.split())
                                    full_content.append(token_text)
                        except json.JSONDecodeError:
                            pass
        else:
            resp = await client.post(url, json=payload, timeout=60.0)
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
            t_after = time.perf_counter()
            data = resp.json()
            tokens_received = data.get("usage", {}).get("completion_tokens", 1)
            ttft = t_after - t0
            full_content.append(data.get("choices", [{}])[0].get("message", {}).get("content", ""))

        total_latency = time.perf_counter() - t0
        if ttft is None:
            ttft = total_latency

        # Calculate Tokens Per Second (TPS)
        tps = (tokens_received / total_latency) if total_latency > 0 else 0.0

        return {
            "request_id": request_id,
            "status": "success",
            "ttft_sec": round(ttft, 4),
            "total_latency_sec": round(total_latency, 4),
            "tokens_generated": tokens_received,
            "tps": round(tps, 2),
        }
    except Exception as exc:
        total_latency = time.perf_counter() - t0
        logger.error("Request #%d failed after %.3fs: %s", request_id, total_latency, exc)
        return {
            "request_id": request_id,
            "status": "failed",
            "error": str(exc),
            "ttft_sec": None,
            "total_latency_sec": round(total_latency, 4),
            "tokens_generated": 0,
            "tps": 0.0,
        }


async def benchmark_inference_concurrency(
    concurrency_levels: List[int] = [5, 10, 20],
    base_url: Optional[str] = None,
    stream: bool = True,
) -> Dict[str, Any]:
    """Execute concurrent inference workloads and report TTFT, TPS, and latency percentiles."""
    logger.info("=================================================================")
    logger.info("STARTING INFERENCE LOAD TEST (POST /v1/chat/completions)")
    logger.info("Levels: %s concurrent requests", concurrency_levels)
    logger.info("=================================================================")

    # Determine transport (live endpoint or ASGI in-process transport)
    is_live_server = False
    if base_url:
        target_base = base_url.rstrip("/")
        is_live_server = True
    else:
        # Check if local dev server on default port 8000 is active
        test_url = "http://127.0.0.1:8000"
        try:
            async with httpx.AsyncClient(timeout=1.0) as chk_client:
                r = await chk_client.get(f"{test_url}/health")
                if r.status_code == 200:
                    target_base = test_url
                    is_live_server = True
                    logger.info("Detected active live serving instance on %s", target_base)
                else:
                    target_base = "http://testserver"
        except Exception:
            target_base = "http://testserver"

    results_by_concurrency = {}

    for c in concurrency_levels:
        logger.info(">>> Executing concurrency tier: %d concurrent requests...", c)
        ram_before = get_system_ram_mb()
        vram_before = get_vram_mb()

        sample_prompts = [
            f"Analyze emergent multi-agent reinforcement learning behavior in scenario #{i}."
            for i in range(c)
        ]

        # Execute concurrent batch
        start_tier = time.perf_counter()

        if is_live_server:
            async with httpx.AsyncClient(timeout=90.0) as client:
                tasks = [
                    _send_single_chat_request(client, target_base, sample_prompts[i], i, stream=stream)
                    for i in range(c)
                ]
                responses = await asyncio.gather(*tasks)
        else:
            # Standalone ASGI in-process execution with real async concurrency
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url=target_base, timeout=90.0) as client:
                tasks = [
                    _send_single_chat_request(client, target_base, sample_prompts[i], i, stream=stream)
                    for i in range(c)
                ]
                responses = await asyncio.gather(*tasks)

        tier_duration = time.perf_counter() - start_tier
        ram_peak = get_system_ram_mb()
        vram_peak = get_vram_mb()

        # Parse metrics
        successful_reqs = [r for r in responses if r["status"] == "success"]
        failed_reqs = [r for r in responses if r["status"] == "failed"]

        ttft_values = [r["ttft_sec"] for r in successful_reqs if r["ttft_sec"] is not None]
        latencies = [r["total_latency_sec"] for r in successful_reqs]
        tps_values = [r["tps"] for r in successful_reqs]
        total_tokens = sum(r["tokens_generated"] for r in successful_reqs)

        aggregate_tps = round(total_tokens / tier_duration, 2) if tier_duration > 0 else 0.0
        ttft_stats = compute_percentiles(ttft_values)
        latency_stats = compute_percentiles(latencies)

        tier_result = {
            "concurrency": c,
            "total_requests": c,
            "successful_requests": len(successful_reqs),
            "failed_requests": len(failed_reqs),
            "wall_clock_duration_sec": round(tier_duration, 3),
            "aggregate_tps": aggregate_tps,
            "total_tokens_generated": total_tokens,
            "ttft_percentiles_sec": ttft_stats,
            "latency_percentiles_sec": latency_stats,
            "tps_stats": compute_percentiles(tps_values),
            "memory_metrics": {
                "ram_baseline_mb": round(ram_before, 2),
                "ram_peak_mb": round(ram_peak, 2),
                "ram_delta_mb": round(ram_peak - ram_before, 2),
                "vram_baseline_mb": round(vram_before, 2),
                "vram_peak_mb": round(vram_peak, 2),
                "vram_delta_mb": round(vram_peak - vram_before, 2),
            },
        }

        results_by_concurrency[f"concurrency_{c}"] = tier_result

        logger.info(
            "Tier %d completed in %.2fs: Aggregate TPS=%.2f, TTFT P50=%.4fs, P95=%.4fs, Latency P50=%.4fs, P95=%.4fs",
            c, tier_duration, aggregate_tps, ttft_stats["p50"], ttft_stats["p95"], latency_stats["p50"], latency_stats["p95"],
        )

    return results_by_concurrency


# ---------------------------------------------------------------------------
# 2. Ingestion & FAISS Vector Daemon Benchmark
# ---------------------------------------------------------------------------

async def benchmark_ingestion_and_vectorization(
    sample_records_count: int = 25,
    settings: Optional[Settings] = None,
    live_harvest: bool = False,
) -> Dict[str, Any]:
    """Measure ingestion throughput (records/sec), embedding throughput, and FAISS index update latency."""
    cfg = settings or get_settings()
    logger.info("=================================================================")
    logger.info("STARTING CONTINUOUS INGESTION & VECTOR MEMORY BENCHMARK")
    logger.info("Sample record batch size: %d", sample_records_count)
    logger.info("=================================================================")

    # Generate synthetic research paper records
    records_data: List[Dict[str, Any]] = []
    for i in range(sample_records_count):
        records_data.append({
            "id": f"bench_arxiv_2026_{i:04d}",
            "title": f"Benchmarking Autonomous Neural Architectures in Distributed Cloud Environments Vol {i}",
            "abstract": (
                f"This benchmark document number {i} examines low-latency streaming inference, "
                "fault-tolerant transactional event storage, and high-concurrency vector retrieval "
                "using quantised open-weights transformer backbones and FAISS IndexFlatIP."
            ),
            "source": "arxiv_benchmark",
            "authors": ["System Benchmark Worker", "Antigravity AI"],
            "categories": ["cs.AI", "cs.DC"],
            "url": f"https://arxiv.org/abs/2609.{i:05d}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # Benchmark 1: Embedding Computation Rate
    logger.info("1. Measuring VectorStore embedding computation rate...")
    bench_mem_dir = cfg.memory_local_dir / "bench_temp"
    bench_mem_dir.mkdir(parents=True, exist_ok=True)
    vector_store = VectorStore(settings=cfg, local_dir=bench_mem_dir)
    embed_start = time.perf_counter()
    texts = [f"{r['title']}\n{r['abstract']}" for r in records_data]
    embeddings = vector_store.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    embed_duration = time.perf_counter() - embed_start

    embed_tps = round(len(records_data) / embed_duration, 2) if embed_duration > 0 else 0.0
    logger.info("Computed %d embeddings in %.3fs (Rate: %.2f embeddings/sec).",
                len(records_data), embed_duration, embed_tps)

    # Benchmark 2: FAISS Index Update Latency
    logger.info("2. Measuring FAISS IndexFlatIP addition & persistence latency...")
    index_start = time.perf_counter()
    added_ids = vector_store.add_documents(records_data, chunk_documents=True)
    index_duration = time.perf_counter() - index_start

    save_start = time.perf_counter()
    saved_index_path = vector_store.save_local()
    save_duration = time.perf_counter() - save_start

    index_ops_per_sec = round(len(added_ids) / index_duration, 2) if index_duration > 0 else 0.0
    logger.info(
        "Indexed %d document chunks into FAISS in %.4fs (Rate: %.2f chunks/sec). Disk save latency: %.4fs",
        len(added_ids), index_duration, index_ops_per_sec, save_duration,
    )

    # Benchmark 3: Continuous Ingestion Pipeline Throughput
    logger.info("3. Measuring continuous ingestion pipeline throughput (records/sec)...")
    daemon = ContinuousIngestionDaemon(settings=cfg)
    daemon.vector_store._embedder = vector_store.embedder
    pipeline_start = time.perf_counter()
    if live_harvest:
        cycle_summary = await daemon.run_once()
        pipeline_duration = time.perf_counter() - pipeline_start
        harvested = cycle_summary.get("harvested", 0)
        status_msg = cycle_summary.get("status", "completed")
    else:
        std_records = [
            StandardizedRecord(
                id=r["id"],
                title=r["title"],
                content=r["abstract"],
                source=r["source"],
                metadata={
                    "authors": r["authors"],
                    "categories": r["categories"],
                    "url": r["url"],
                },
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            for r in records_data
        ]
        # Stage to JSONL
        staging_file = cfg.staging_dir / f"bench_harvest_{int(time.time())}.jsonl"
        daemon.ingestion_client.save_records_to_jsonl(std_records, staging_file)
        # Audit into MySQL manager (relational storage / resilient buffer)
        for r in std_records:
            daemon.mysql_mgr.record_paper(r.model_dump())
        pipeline_duration = time.perf_counter() - pipeline_start
        harvested = len(std_records)
        status_msg = "completed"

    ingestion_rate = round(harvested / pipeline_duration, 2) if pipeline_duration > 0 and harvested > 0 else 0.0
    logger.info("Ingestion pipeline processed %d records in %.4fs (Rate: %.2f records/sec).",
                harvested, pipeline_duration, ingestion_rate)

    return {
        "embedding_benchmark": {
            "documents_count": sample_records_count,
            "duration_sec": round(embed_duration, 4),
            "embeddings_per_sec": embed_tps,
            "embedding_dim": embeddings.shape[1],
        },
        "faiss_index_benchmark": {
            "chunks_indexed": len(added_ids),
            "insertion_duration_sec": round(index_duration, 4),
            "insertion_rate_chunks_per_sec": index_ops_per_sec,
            "persistence_duration_sec": round(save_duration, 4),
            "index_path": str(saved_index_path),
            "total_vectors_in_index": vector_store.index.ntotal,
        },
        "daemon_cycle_benchmark": {
            "status": status_msg,
            "harvested_count": harvested,
            "total_cycle_duration_sec": round(pipeline_duration, 3),
            "ingestion_rate_records_per_sec": ingestion_rate,
        },
    }


# ---------------------------------------------------------------------------
# Formatting & CLI Entrypoint
# ---------------------------------------------------------------------------

def print_summary_tables(
    inference_results: Dict[str, Any],
    ingest_results: Dict[str, Any],
) -> str:
    """Render structured ASCII / Markdown tables for benchmarks and return report string."""
    report_lines = []
    report_lines.append("# Operational Benchmarks & Performance Validation Report")
    report_lines.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    report_lines.append("")

    report_lines.append("## 1. Inference Engine Concurrency Benchmark (`POST /v1/chat/completions`)")
    report_lines.append("")
    report_lines.append("| Concurrency | Aggregate TPS | TTFT P50 (s) | TTFT P95 (s) | TTFT P99 (s) | Latency P50 (s) | Latency P95 (s) | Latency P99 (s) | RAM Peak (MB) | VRAM Peak (MB) |")
    report_lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for key, data in inference_results.items():
        c = data["concurrency"]
        tps = data["aggregate_tps"]
        ttft = data["ttft_percentiles_sec"]
        lat = data["latency_percentiles_sec"]
        mem = data["memory_metrics"]
        report_lines.append(
            f"| {c:2d} workers | {tps:13.2f} | {ttft['p50']:12.4f} | {ttft['p95']:12.4f} | {ttft['p99']:12.4f} | "
            f"{lat['p50']:15.4f} | {lat['p95']:15.4f} | {lat['p99']:15.4f} | {mem['ram_peak_mb']:13.2f} | {mem['vram_peak_mb']:14.2f} |"
        )

    report_lines.append("")
    report_lines.append("## 2. Ingestion Daemon & FAISS Vector Memory Benchmark")
    report_lines.append("")
    report_lines.append("| Subsystem Component | Batch Size | Duration (s) | Throughput Rate | Index State |")
    report_lines.append("|---|---|---|---|---|")

    emb = ingest_results["embedding_benchmark"]
    fidx = ingest_results["faiss_index_benchmark"]
    dmn = ingest_results["daemon_cycle_benchmark"]

    report_lines.append(f"| SentenceTransformer Encoding | {emb['documents_count']} docs | {emb['duration_sec']}s | {emb['embeddings_per_sec']} embeddings/s | Dim: {emb['embedding_dim']} |")
    report_lines.append(f"| FAISS IndexFlatIP Chunk Append | {fidx['chunks_indexed']} chunks | {fidx['insertion_duration_sec']}s | {fidx['insertion_rate_chunks_per_sec']} chunks/s | Total: {fidx['total_vectors_in_index']} |")
    report_lines.append(f"| Continuous Ingest Run-Once Cycle | {dmn['harvested_count']} harvested | {dmn['total_cycle_duration_sec']}s | {dmn['ingestion_rate_records_per_sec']} records/s | Status: {dmn['status']} |")
    report_lines.append("")

    full_report = "\n".join(report_lines)
    print("\n" + full_report + "\n")
    return full_report


async def main_async() -> None:
    parser = argparse.ArgumentParser(description="Run complete load tests and performance benchmarks.")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[5, 10, 20],
                        help="Concurrency worker counts (default: 5 10 20)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="Base URL of running inference service (e.g. http://127.0.0.1:8000)")
    parser.add_argument("--ingest-samples", type=int, default=25,
                        help="Number of synthetic records to benchmark for vectorization (default: 25)")
    parser.add_argument("--out-dir", type=str, default="benchmarks",
                        help="Output directory to save metrics JSON and Markdown report")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming response mode for inference requests")
    parser.add_argument("--live-harvest", action="store_true",
                        help="Execute live remote harvesting from ArXiv/RSS instead of fast synthetic pipeline")
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Run inference concurrency benchmark
    inference_metrics = await benchmark_inference_concurrency(
        concurrency_levels=args.concurrency,
        base_url=args.base_url,
        stream=not args.no_stream,
    )

    # 2. Run ingestion & vectorization benchmark
    ingest_metrics = await benchmark_ingestion_and_vectorization(
        sample_records_count=args.ingest_samples,
        live_harvest=args.live_harvest,
    )

    # 3. Print report and save outputs
    report_md = print_summary_tables(inference_metrics, ingest_metrics)

    md_file = out_path / "benchmark_report.md"
    md_file.write_text(report_md, encoding="utf-8")

    json_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "inference_concurrency_benchmark": inference_metrics,
        "ingestion_vector_benchmark": ingest_metrics,
    }
    json_file = out_path / "benchmark_results.json"
    json_file.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    report_json = out_path / "benchmark_report.json"
    report_json.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    logger.info("Saved benchmark reports to %s, %s, and %s", md_file, json_file, report_json)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
