"""Configuration management for GCS integration and stream ingestion subsystem.

Leverages Pydantic v2 BaseSettings to load and validate environment variables,
supporting .env files and production system environments.
"""

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Subsystem configuration loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # --- Cloud Storage Configuration ---
    gcs_bucket_name: str = Field(
        default="agi-agent-ingestion-data",
        description="Target Google Cloud Storage bucket name for persistent records.",
        validation_alias="GCS_BUCKET_NAME",
    )
    gcs_credentials_path: Optional[str] = Field(
        default=None,
        description="Path to Google Cloud service account JSON credentials file.",
        validation_alias="GCS_CREDENTIALS_PATH",
    )
    gcs_chunk_size: int = Field(
        default=8 * 1024 * 1024,  # 8 MB
        description="Chunk buffer size in bytes for streaming uploads and downloads (must be a multiple of 256 KiB).",
        validation_alias="GCS_CHUNK_SIZE",
    )
    local_lake_dir: Path = Field(
        default=Path("./data_storage/local_lake"),
        description="Local disk storage lake fallback directory when remote GCS credentials are not present.",
        validation_alias="LOCAL_LAKE_DIR",
    )

    # --- HTTP & Ingestion Configuration ---
    user_agent: str = Field(
        default="AGI-Agent-Ingestion/1.0 (+https://github.com/Rajuzet/AGI-pack; contact=autonomous-agent@cloud.internal)",
        description="Standardized User-Agent header for external HTTP/RSS requests.",
        validation_alias="USER_AGENT",
    )
    poll_interval_seconds: int = Field(
        default=300,
        ge=5,
        description="Default polling interval in seconds between stream ingestion cycles.",
        validation_alias="POLL_INTERVAL_SECONDS",
    )
    http_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Request timeout in seconds for async HTTP clients.",
        validation_alias="HTTP_TIMEOUT_SECONDS",
    )

    # --- Source Specifics: ArXiv ---
    arxiv_api_url: str = Field(
        default="https://export.arxiv.org/api/query",
        description="Base URL for ArXiv Atom Query API.",
        validation_alias="ARXIV_API_URL",
    )
    arxiv_categories: List[str] = Field(
        default_factory=lambda: ["cs.AI", "cs.LG", "stat.ML"],
        description="Default scientific categories queried from ArXiv.",
        validation_alias="ARXIV_CATEGORIES",
    )
    arxiv_max_results: int = Field(
        default=25,
        ge=1,
        le=500,
        description="Maximum number of papers retrieved per ArXiv request.",
        validation_alias="ARXIV_MAX_RESULTS",
    )

    # --- Source Specifics: Authentic Real-Time RSS ---
    rss_feeds: Dict[str, str] = Field(
        default_factory=lambda: {
            "nature_news": "https://www.nature.com/nature.rss",
            "mit_tech_review": "https://www.technologyreview.com/feed/",
            "arxiv_cs_ai_latest": "https://rss.arxiv.org/rss/cs.AI",
        },
        description="Mapping of authentic RSS feed identifiers to valid remote feed URLs.",
    )

    # --- Text Normalization & Limiting ---
    max_content_tokens: int = Field(
        default=4000,
        ge=50,
        description="Estimated upper token bound for normalized record content.",
        validation_alias="MAX_CONTENT_TOKENS",
    )

    # --- Storage & Staging Paths ---
    staging_dir: Path = Field(
        default=Path("./data_staging"),
        description="Local directory for temporary JSONL buffers and chunk staging.",
        validation_alias="STAGING_DIR",
    )

    # --- Vector Memory Engine Configuration ---
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformers model name or HuggingFace repo ID for dense embeddings.",
        validation_alias="EMBEDDING_MODEL_NAME",
    )
    vector_dimension: int = Field(
        default=384,
        ge=1,
        description="Dimension of the embedding vectors (384 for all-MiniLM-L6-v2).",
        validation_alias="VECTOR_DIMENSION",
    )
    chunk_max_tokens: int = Field(
        default=512,
        ge=50,
        description="Target maximum token count per semantic document chunk.",
        validation_alias="CHUNK_MAX_TOKENS",
    )
    chunk_overlap_tokens: int = Field(
        default=50,
        ge=0,
        description="Token overlap between consecutive semantic chunks.",
        validation_alias="CHUNK_OVERLAP_TOKENS",
    )
    memory_local_dir: Path = Field(
        default=Path("./data_memory"),
        description="Local directory for persistent FAISS index and metadata store.",
        validation_alias="MEMORY_LOCAL_DIR",
    )
    memory_remote_prefix: str = Field(
        default="vector_memory/latest",
        description="Target GCS folder prefix for long-term index synchronization.",
        validation_alias="MEMORY_REMOTE_PREFIX",
    )

    # --- Relational Audit Database (MySQL) ---
    mysql_host: str = Field(
        default="localhost",
        description="MySQL server hostname or Cloud SQL IP address.",
        validation_alias="MYSQL_HOST",
    )
    mysql_port: int = Field(
        default=3306,
        ge=1,
        le=65535,
        description="MySQL server connection port.",
        validation_alias="MYSQL_PORT",
    )
    mysql_user: str = Field(
        default="root",
        description="MySQL username for authentication.",
        validation_alias="MYSQL_USER",
    )
    mysql_password: str = Field(
        default="",
        description="MySQL user password.",
        validation_alias="MYSQL_PASSWORD",
    )
    mysql_database: str = Field(
        default="agi_memory",
        description="Target MySQL database schema name for audit logs.",
        validation_alias="MYSQL_DATABASE",
    )
    mysql_pool_size: int = Field(
        default=5,
        ge=1,
        le=32,
        description="Connection pool size for concurrent database queries.",
        validation_alias="MYSQL_POOL_SIZE",
    )

    # --- Serving Microservice Configuration ---
    serving_host: str = Field(
        default="0.0.0.0",
        description="Host address for model serving microservice.",
        validation_alias="SERVING_HOST",
    )
    serving_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Port for model serving microservice.",
        validation_alias="SERVING_PORT",
    )
    serving_model_id: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="Base model identifier for serving.",
        validation_alias="SERVING_MODEL_ID",
    )
    serving_gcs_adapter_prefix: str = Field(
        default="models/lora_checkpoints/latest",
        description="GCS path prefix for PEFT LoRA adapter checkpoints.",
        validation_alias="SERVING_GCS_ADAPTER_PREFIX",
    )

    # --- Continuous Ingestion Daemon ---
    daemon_poll_interval: int = Field(
        default=300,
        ge=5,
        description="Interval in seconds between continuous ingestion harvest cycles.",
        validation_alias="DAEMON_POLL_INTERVAL",
    )
    daemon_max_records_per_poll: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Max papers/articles harvested per source per poll cycle.",
        validation_alias="DAEMON_MAX_RECORDS_PER_POLL",
    )

    # --- Observability ---
    log_level: str = Field(
        default="INFO",
        description="Application logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
        validation_alias="LOG_LEVEL",
    )

    # --- Self-Evolution & Prompt Optimization ---
    system_prompts_config_path: str = Field(
        default="agent/system_prompts.json",
        description="Path to persisted winning system prompts configuration.",
        validation_alias="SYSTEM_PROMPTS_CONFIG_PATH",
    )
    curated_evolution_pairs_path: str = Field(
        default="data_staging/curated_evolution_pairs.jsonl",
        description="Path to curated instruction-tuning pairs for model retraining.",
        validation_alias="CURATED_EVOLUTION_PAIRS_PATH",
    )
    auto_retrain_sample_threshold: int = Field(
        default=500,
        description="Minimum new curated samples required to trigger autonomous retraining.",
        validation_alias="AUTO_RETRAIN_SAMPLE_THRESHOLD",
    )
    auto_retrain_score_threshold: float = Field(
        default=70.0,
        description="Benchmark composite score below which retraining is automatically triggered.",
        validation_alias="AUTO_RETRAIN_SCORE_THRESHOLD",
    )

    @field_validator("gcs_chunk_size")
    @classmethod
    def validate_chunk_size_alignment(cls, value: int) -> int:
        """GCS requires chunk transfers to be multiples of 256 KiB (262,144 bytes)."""
        alignment = 256 * 1024
        if value < alignment or value % alignment != 0:
            raise ValueError(
                f"gcs_chunk_size must be a positive multiple of 256 KiB ({alignment} bytes). Got {value}."
            )
        return value

    @field_validator("gcs_credentials_path", mode="before")
    @classmethod
    def validate_credentials_path(cls, value: Optional[str]) -> Optional[str]:
        """Validate and resolve credentials path from GCS_CREDENTIALS_PATH or GOOGLE_APPLICATION_CREDENTIALS."""
        if value:
            expanded = os.path.expanduser(os.path.expandvars(value))
            return expanded

        # Fallback to standard GOOGLE_APPLICATION_CREDENTIALS environment variable
        gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if gac:
            expanded = os.path.expanduser(os.path.expandvars(gac))
            return expanded

        return value

    def load_active_system_prompt(self) -> Optional[str]:
        """Load the active system prompt template from system_prompts_config_path if available."""
        path = Path(self.system_prompts_config_path)
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return data.get("rendered_system_prompt")
            except Exception:
                return None
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Obtain a cached, validated singleton instance of Settings."""
    return Settings()
