# Operational Benchmarks & Performance Validation Report
**Generated:** 2026-09-21T16:38:15.160608+00:00

## 1. Inference Engine Concurrency Benchmark (`POST /v1/chat/completions`)

| Concurrency | Aggregate TPS | TTFT P50 (s) | TTFT P95 (s) | TTFT P99 (s) | Latency P50 (s) | Latency P95 (s) | Latency P99 (s) | RAM Peak (MB) | VRAM Peak (MB) |
|---|---|---|---|---|---|---|---|---|---|
|  5 workers |        611.48 |       0.1406 |       0.1496 |       0.1496 |          0.2262 |          0.2294 |          0.2294 |        374.49 |           0.00 |
| 10 workers |        962.17 |       0.1328 |       0.1666 |       0.1666 |          0.2819 |          0.3164 |          0.3164 |        375.79 |           0.00 |

## 2. Ingestion Daemon & FAISS Vector Memory Benchmark

| Subsystem Component | Batch Size | Duration (s) | Throughput Rate | Index State |
|---|---|---|---|---|
| SentenceTransformer Encoding | 10 docs | 15.1899s | 0.66 embeddings/s | Dim: 384 |
| FAISS IndexFlatIP Chunk Append | 10 chunks | 0.4758s | 21.02 chunks/s | Total: 10 |
| Continuous Ingest Run-Once Cycle | 10 harvested | 1.243s | 8.05 records/s | Status: completed |
