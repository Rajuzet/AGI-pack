"""Production Prometheus metrics collectors and instrumentation for AGI Autonomous Agent.

Collects telemetry across model serving, continuous ingestion, vector memory,
ReAct reasoning cycles, and system resources.
"""

from contextlib import contextmanager
import logging
import os
import time
from typing import Any, Callable, Dict, Generator, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    REGISTRY,
    generate_latest,
)
import psutil
import torch

logger = logging.getLogger("agi_monitoring.metrics")

# Use standard or dedicated registry
REGISTRY_NAME = "agi_registry"
METRICS_REGISTRY: CollectorRegistry = REGISTRY


# ---------------------------------------------------------------------------
# 1. Counters
# ---------------------------------------------------------------------------

PAPERS_INGESTED_TOTAL = Counter(
    "agi_papers_ingested_total",
    "Total number of research papers and articles ingested into persistent memory.",
    ["domain"],
    registry=METRICS_REGISTRY,
)

TOOL_EXECUTION_CALLS_TOTAL = Counter(
    "agi_tool_execution_calls_total",
    "Total number of agent tool invocations categorized by tool name and status.",
    ["tool_name", "status"],
    registry=METRICS_REGISTRY,
)

MODEL_RELOAD_ATTEMPTS_TOTAL = Counter(
    "agi_model_reload_attempts_total",
    "Total number of PEFT LoRA adapter hot-reload attempts.",
    ["status"],
    registry=METRICS_REGISTRY,
)


# ---------------------------------------------------------------------------
# 2. Histograms
# ---------------------------------------------------------------------------

REACT_STEP_LATENCY_SECONDS = Histogram(
    "agi_react_step_latency_seconds",
    "Latency distribution of individual ReAct reasoning cycles in seconds.",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0],
    registry=METRICS_REGISTRY,
)

TIME_TO_FIRST_TOKEN_SECONDS = Histogram(
    "agi_time_to_first_token_seconds",
    "Time to First Token (TTFT) latency for conversational generation in seconds.",
    buckets=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0],
    registry=METRICS_REGISTRY,
)

TOKENS_PER_SECOND = Histogram(
    "agi_tokens_per_second",
    "Generation throughput distribution in tokens per second (TPS).",
    buckets=[10, 25, 50, 100, 200, 400, 800, 1200, 2000],
    registry=METRICS_REGISTRY,
)

FAISS_QUERY_LATENCY_SECONDS = Histogram(
    "agi_faiss_query_latency_seconds",
    "Latency distribution of dense FAISS cosine vector searches in seconds.",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    registry=METRICS_REGISTRY,
)


# ---------------------------------------------------------------------------
# 3. Gauges
# ---------------------------------------------------------------------------

ACTIVE_MEMORY_VECTORS = Gauge(
    "agi_active_memory_vectors",
    "Total count of active document vectors indexed in local FAISS memory.",
    registry=METRICS_REGISTRY,
)

MYSQL_BUFFER_QUEUE_SIZE = Gauge(
    "agi_mysql_buffer_queue_size",
    "Number of unwritten audit records queued in-memory during database disconnection.",
    registry=METRICS_REGISTRY,
)

SYSTEM_RAM_USAGE_BYTES = Gauge(
    "agi_system_ram_usage_bytes",
    "Host system resident memory (RSS) consumed by the running process in bytes.",
    registry=METRICS_REGISTRY,
)

SYSTEM_VRAM_USAGE_BYTES = Gauge(
    "agi_system_vram_usage_bytes",
    "GPU Video RAM (VRAM) allocated by PyTorch CUDA in bytes.",
    registry=METRICS_REGISTRY,
)


# ---------------------------------------------------------------------------
# Telemetry Helpers & Context Managers
# ---------------------------------------------------------------------------

_LAST_GAUGE_UPDATE: float = 0.0
_CACHED_RAM_BYTES: float = 0.0
_CACHED_VRAM_BYTES: float = 0.0


def update_system_gauges(
    active_vectors: Optional[int] = None,
    buffered_count: Optional[int] = None,
) -> Dict[str, float]:
    """Sample and update host resource and store metrics with 1.0s TTL throttling."""
    global _LAST_GAUGE_UPDATE, _CACHED_RAM_BYTES, _CACHED_VRAM_BYTES
    now = time.time()
    if now - _LAST_GAUGE_UPDATE > 1.0 or _CACHED_RAM_BYTES == 0.0:
        _LAST_GAUGE_UPDATE = now
        # RAM
        try:
            process = psutil.Process()
            _CACHED_RAM_BYTES = float(process.memory_info().rss)
            SYSTEM_RAM_USAGE_BYTES.set(_CACHED_RAM_BYTES)
        except Exception as exc:
            logger.debug("Failed to read process RAM: %s", exc)
            _CACHED_RAM_BYTES = 0.0

        # VRAM
        if torch.cuda.is_available():
            try:
                _CACHED_VRAM_BYTES = float(torch.cuda.memory_allocated())
                SYSTEM_VRAM_USAGE_BYTES.set(_CACHED_VRAM_BYTES)
            except Exception as exc:
                logger.debug("Failed to read CUDA VRAM: %s", exc)
        else:
            SYSTEM_VRAM_USAGE_BYTES.set(0.0)
            _CACHED_VRAM_BYTES = 0.0

    ram_bytes = _CACHED_RAM_BYTES
    vram_bytes = _CACHED_VRAM_BYTES

    # Vector store & MySQL buffer
    if active_vectors is not None:
        ACTIVE_MEMORY_VECTORS.set(float(active_vectors))
    if buffered_count is not None:
        MYSQL_BUFFER_QUEUE_SIZE.set(float(buffered_count))

    return {
        "ram_bytes": ram_bytes,
        "vram_bytes": vram_bytes,
        "active_vectors": float(active_vectors or 0),
        "buffered_count": float(buffered_count or 0),
    }


def record_paper_ingested(domain: str, count: int = 1) -> None:
    """Increment papers ingested counter for a specific domain."""
    PAPERS_INGESTED_TOTAL.labels(domain=domain).inc(count)


def record_tool_call(tool_name: str, success: bool = True) -> None:
    """Record an agent tool execution status."""
    status_label = "success" if success else "failure"
    TOOL_EXECUTION_CALLS_TOTAL.labels(tool_name=tool_name, status=status_label).inc()


def record_model_reload(status: str) -> None:
    """Record model reload attempt outcome (e.g., 'success', 'corrupted_rollback', 'gcs_missing')."""
    MODEL_RELOAD_ATTEMPTS_TOTAL.labels(status=status).inc()


@contextmanager
def time_react_step() -> Generator[None, None, None]:
    """Context manager measuring ReAct cycle latency."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        REACT_STEP_LATENCY_SECONDS.observe(elapsed)


@contextmanager
def time_faiss_query() -> Generator[None, None, None]:
    """Context manager measuring FAISS vector search latency."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        FAISS_QUERY_LATENCY_SECONDS.observe(elapsed)


def get_latest_metrics() -> bytes:
    """Generate Prometheus exposition format payload."""
    update_system_gauges()
    return generate_latest(METRICS_REGISTRY)
