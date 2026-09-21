# Autonomous Agent Infrastructure: Production Operational Runbook

**Version:** 1.0.0  
**Classification:** Production Engineering & SRE Runbook  
**Target Environment:** Containerized Docker Compose, Google Cloud Storage, MySQL 8.0, and PEFT Model Serving.

---

## 1. System Architecture & Topology Overview

The system operates as an autonomous, self-healing, distributed AI pipeline comprised of three decoupled containerized layers:

```
                  +----------------------------------------------+
                  |         Client / Evaluator / Agent           |
                  +----------------------------------------------+
                                         |
                                         v
       +--------------------------------------------------------------------+
       |                   Docker Network: agi-production                   |
       |                                                                    |
       |  +------------------------+        +----------------------------+  |
       |  |  agi-serving (8000)    | <----> |   agi-mysql (3306)         |  |
       |  |  - FastAPI + PEFT/vLLM |        |   - Relational Audit DB    |  |
       |  |  - Hot-Reload from GCS |        |   - In-Memory Buffering    |  |
       |  +------------------------+        +----------------------------+  |
       |               ^                                   ^                |
       |               |                                   |                |
       |  +------------------------+                       |                |
       |  |  agi-daemon            | ----------------------+                |
       |  |  - ArXiv / RSS Harvest |                                        |
       |  |  - FAISS Vector Memory |                                        |
       |  +------------------------+                                        |
       +--------------------------------------------------------------------+
                         |                                 |
                         v                                 v
          +-----------------------------------------------------------------+
          |    Cloud Persistence Layer (Google Cloud Storage Bucket)        |
          |    - Data Lake: gs://<bucket>/ingestion_stream/YYYYMMDD/        |
          |    - Vector Memory: gs://<bucket>/vector_memory/latest/         |
          |    - LoRA Adapters: gs://<bucket>/models/lora_checkpoints/     |
          +-----------------------------------------------------------------+
```

---

## 2. Production Startup & Stack Lifecycle

### 2.1 Pre-Flight Environment Configuration
Create or verify your production `.env` configuration file in the project root:

```bash
# Core Cloud Storage Configuration
GCS_BUCKET_NAME=agi-agent-ingestion-data
GCS_CREDENTIALS_PATH=/app/secrets/gcs_service_account.json
GCS_CHUNK_SIZE=8388608

# Relational Database Configuration
MYSQL_HOST=agi-mysql
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=agi_root_secure_prod_2026
MYSQL_DATABASE=agi_memory
MYSQL_POOL_SIZE=10

# Continuous Ingestion Daemon Settings
DAEMON_POLL_INTERVAL=300
DAEMON_MAX_RECORDS_PER_POLL=25
ARXIV_CATEGORIES=["cs.AI", "cs.LG", "stat.ML"]

# Model Serving Settings
SERVING_HOST=0.0.0.0
SERVING_PORT=8000
SERVING_MODEL_ID=Qwen/Qwen2.5-7B-Instruct
SERVING_GCS_ADAPTER_PREFIX=models/lora_checkpoints/latest
```

### 2.2 Launching the Production Stack
Execute the following command to build, start, and run all services in detached mode:

```bash
docker compose -f deployment/docker-compose.yml up -d --build
```

### 2.3 Validating Stack Initialization
Inspect service status and healthcheck telemetry:

```bash
# Check container status
docker compose -f deployment/docker-compose.yml ps

# Confirm MySQL health
docker exec agi-mysql mysqladmin ping -h localhost -u root -p$MYSQL_PASSWORD

# Confirm Model Serving health and active adapter
curl -s http://localhost:8000/health | jq .
```

Expected output for `/health`:
```json
{
  "status": "healthy",
  "warmed_up": true,
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "active_adapter": null,
  "active_adapter_meta": {},
  "backend": "peft-hf",
  "device": "cuda",
  "vram_allocated_gb": 4.82,
  "vram_reserved_gb": 6.10,
  "timestamp": "2026-09-20T14:30:00.000000+00:00"
}
```

### 2.4 Automated Stack Smoke Testing
Run the ephemeral smoke test suite within the dedicated test network:

```bash
docker compose -f deployment/docker-compose.test.yml up --abort-on-container-exit --build
```

---

## 3. Continuous Log Inspection & Observability

### 3.1 Ingestion Daemon Harvesting Logs
Monitor real-time harvesting cycles, ArXiv XML parsing, RSS ingestion, deduplication, and FAISS vectorization:

```bash
# Tail daemon harvesting logs
docker compose -f deployment/docker-compose.yml logs -f agi-daemon

# Grep for harvesting and vector indexing throughput
docker compose -f deployment/docker-compose.yml logs agi-daemon | grep -E "(Harvested|Indexed|Audited)"
```

### 3.2 Model Inference & Serving Logs
Inspect incoming conversational completions, streaming chunks, latency, and adapter swap triggers:

```bash
# Tail inference server logs
docker compose -f deployment/docker-compose.yml logs -f agi-serving

# Filter specifically for adapter hot-reloads and generation calls
docker compose -f deployment/docker-compose.yml logs agi-serving | grep -E "(reload|chat/completions|VRAM)"
```

### 3.3 Relational Audit Records Inspection (MySQL)
Query audit logs directly from the running container:

```bash
# Inspect the 10 most recently ingested research papers
docker exec -it agi-mysql mysql -u root -p$MYSQL_PASSWORD agi_memory -e "
  SELECT id, source, title, ingested_at 
  FROM research_papers 
  ORDER BY ingested_at DESC 
  LIMIT 10;
"

# Inspect completed agent reasoning sessions and latency metrics
docker exec -it agi-mysql mysql -u root -p$MYSQL_PASSWORD agi_memory -e "
  SELECT session_id, status, total_steps, execution_latency, session_timestamp 
  FROM agent_sessions 
  ORDER BY session_timestamp DESC 
  LIMIT 10;
"

# Inspect fine-tuning training runs and loss telemetry
docker exec -it agi-mysql mysql -u root -p$MYSQL_PASSWORD agi_memory -e "
  SELECT run_id, base_model_id, quantization_type, final_loss, gcs_checkpoint_uri, executed_at 
  FROM training_runs 
  ORDER BY executed_at DESC 
  LIMIT 5;
"
```

### 3.4 Live Container Resource Metrics
```bash
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}\t{{.NetIO}}\t{{.BlockIO}}"
```

---

## 4. Zero-Downtime Adapter Deployment (LoRA Hot-Reload)

The system allows continuous fine-tuning on external GPU runtimes (such as free-tier Kaggle or Google Colab) with instantaneous, zero-downtime deployment to live serving containers.

### Step 1: Execute LoRA Fine-Tuning
In Kaggle or Google Colab, execute `training/kaggle_lora_train.py` with your Kaggle GPU:
- Trains quantized 4-bit NF4 PEFT adapter on harvested JSONL datasets.
- Streams resulting artifacts (`adapter_model.safetensors`, `adapter_config.json`, `training_run.json`) directly into Google Cloud Storage at prefix:
  `gs://agi-agent-ingestion-data/models/lora_checkpoints/<run_id>/`
- Automatically updates `gs://agi-agent-ingestion-data/models/lora_checkpoints/latest/` pointer.

### Step 2: Trigger Zero-Downtime Live Reload
Send an authorized `POST` request to `/v1/models/reload`:

```bash
curl -X POST http://localhost:8000/v1/models/reload \
  -H "Content-Type: application/json" \
  -d '{"gcs_prefix": "models/lora_checkpoints/latest"}'
```

Response:
```json
{
  "status": "success",
  "message": "Successfully hot-swapped LoRA adapter from gs://agi-agent-ingestion-data/models/lora_checkpoints/latest",
  "active_adapter": "latest",
  "gcs_prefix": "models/lora_checkpoints/latest",
  "timestamp": "2026-09-20T14:45:00.000000+00:00"
}
```

### Step 3: Verify Rollback Protection on Corrupted Payloads
If a checkpoint in GCS has missing weights, 0-byte files, or malformed JSON configs:
1. `POST /v1/models/reload` catches the error before modifying the active PyTorch model.
2. Returns HTTP 500 (`"Corrupted adapter payload: ... Rolled back to active adapter: <prev_name>"`).
3. The previous model adapter remains active in memory.
4. The service continues serving `/v1/chat/completions` without dropped connections or service restarts.

---

## 5. Backup & Disaster Recovery Procedures

### 5.1 FAISS Vector Memory Backup & Recovery

#### Automatic Backup:
The `ContinuousIngestionDaemon` autonomously persists local vectors to disk and streams the index to GCS every cycle:
- `data_memory/index.faiss`
- `data_memory/metadata.json`
- Destination: `gs://<bucket>/vector_memory/latest/`

#### Manual Snapshot Backup:
```bash
# Create local timestamped archive
DATE_TAG=$(date +%Y%m%d_%H%M%S)
tar -czvf backups/faiss_backup_${DATE_TAG}.tar.gz -C data_memory index.faiss metadata.json

# Stream snapshot to GCS
gcloud storage cp backups/faiss_backup_${DATE_TAG}.tar.gz gs://${GCS_BUCKET_NAME}/backups/vector_memory/
```

#### Vector Index Disaster Recovery:
To restore vector memory onto a fresh or corrupted node:
```bash
# 1. Download latest index snapshot from GCS
gcloud storage cp gs://${GCS_BUCKET_NAME}/vector_memory/latest/index.faiss data_memory/index.faiss
gcloud storage cp gs://${GCS_BUCKET_NAME}/vector_memory/latest/metadata.json data_memory/metadata.json

# 2. Restart daemon to load index
docker compose -f deployment/docker-compose.yml restart agi-daemon
```

---

### 5.2 MySQL Relational Audit Database Backup & Restore

#### Hot Logical Backup (mysqldump):
```bash
# Create backup directory
mkdir -p backups

# Perform atomic, consistent logical dump
BACKUP_FILE="backups/agi_memory_$(date +%Y%m%d_%H%M%S).sql"
docker exec agi-mysql mysqldump \
  -u root \
  -p${MYSQL_PASSWORD} \
  --single-transaction \
  --quick \
  --databases agi_memory > ${BACKUP_FILE}

echo "Database successfully backed up to ${BACKUP_FILE} ($(wc -c < ${BACKUP_FILE}) bytes)"
```

#### Point-in-Time Restore:
```bash
# Restore schema and records into running MySQL container
TARGET_BACKUP="backups/agi_memory_restore.sql"
docker exec -i agi-mysql mysql -u root -p${MYSQL_PASSWORD} agi_memory < ${TARGET_BACKUP}
```

#### In-Memory Buffer Reconnect Recovery:
If MySQL experiences an unplanned network partition or outage during runtime:
1. `MySQLAuditManager` detects connection failure and enters non-blocking offline mode within 3 seconds.
2. All subsequent paper, session, and step records are captured in thread-safe memory queues (`_buffered_papers`, `_buffered_sessions`, `_buffered_training_runs`).
3. Upon database reconnection, `mysql_mgr.flush_buffer()` commits all accumulated records in transaction batches with zero data loss.

---

## 6. Performance Benchmarking & Load Testing

Run the standardized benchmarking harness to validate inference latency, concurrency throughput, and vectorization rates:

```bash
# Run complete test across 5, 10, and 20 concurrent requests
python benchmarks/run_load_test.py --concurrency 5 10 20 --ingest-samples 25

# Run against a live running production endpoint
python benchmarks/run_load_test.py --base-url http://localhost:8000 --concurrency 5 10 20
```

### Key Performance Targets:
- **Inference Time to First Token (TTFT):** P50 < 1.0s, P95 < 1.5s
- **Inference Concurrency Scaling:** $\ge$ 150 TPS (5 workers), $\ge$ 300 TPS (10 workers), $\ge$ 800 TPS (20 workers)
- **Ingestion Throughput:** $\ge$ 10 records/sec
- **FAISS Vector Index Insertion:** $\ge$ 30 chunks/sec
