"""Stream ingestion package initialization."""

from ingestion.stream_sources import (
    StandardizedRecord,
    StreamIngestionClient,
    TextNormalizer,
)

__all__ = ["StandardizedRecord", "StreamIngestionClient", "TextNormalizer"]
