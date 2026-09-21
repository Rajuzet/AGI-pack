# =============================================================================
# Stage 1: Build & Dependency Wheel Stage
# =============================================================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .

# Upgrade pip and install production wheels
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Hardened Production Runtime Image
# =============================================================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-built virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create unprivileged application user and staging directories
RUN groupadd -g 10001 agi && \
    useradd -u 10001 -g agi -s /bin/bash -m agiuser && \
    mkdir -p /app/data_staging /app/data_memory /app/model_checkpoints && \
    chown -R agiuser:agi /app

WORKDIR /app

# Copy application source code
COPY --chown=agiuser:agi config.py main.py requirements.txt ./
COPY --chown=agiuser:agi agent/ ./agent/
COPY --chown=agiuser:agi benchmarks/ ./benchmarks/
COPY --chown=agiuser:agi daemon/ ./daemon/
COPY --chown=agiuser:agi ingestion/ ./ingestion/
COPY --chown=agiuser:agi memory/ ./memory/
COPY --chown=agiuser:agi monitoring/ ./monitoring/
COPY --chown=agiuser:agi optimization/ ./optimization/
COPY --chown=agiuser:agi self_evolution/ ./self_evolution/
COPY --chown=agiuser:agi serving/ ./serving/
COPY --chown=agiuser:agi storage/ ./storage/
COPY --chown=agiuser:agi training/ ./training/

USER agiuser

# Expose model serving port
EXPOSE 8000

# Health check against serving endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entrypoint starts the FastAPI inference microservice
CMD ["python3", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
