"""Unit tests for configuration validation and environment loading."""

import os
import pytest
from pydantic import ValidationError

from config import Settings, get_settings


def test_default_settings():
    """Verify production default settings load correctly."""
    settings = Settings()
    assert settings.gcs_bucket_name == "agi-agent-ingestion-data"
    assert settings.gcs_chunk_size == 8 * 1024 * 1024
    assert settings.poll_interval_seconds == 300
    assert "cs.AI" in settings.arxiv_categories
    assert "nature_news" in settings.rss_feeds
    assert settings.max_content_tokens == 4000


def test_custom_env_override(monkeypatch):
    """Verify environment variables properly override default configuration."""
    monkeypatch.setenv("GCS_BUCKET_NAME", "custom-agi-bucket")
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "600")
    monkeypatch.setenv("MAX_CONTENT_TOKENS", "2048")
    monkeypatch.setenv("USER_AGENT", "CustomAgent/2.0")

    settings = Settings()
    assert settings.gcs_bucket_name == "custom-agi-bucket"
    assert settings.poll_interval_seconds == 600
    assert settings.max_content_tokens == 2048
    assert settings.user_agent == "CustomAgent/2.0"


def test_chunk_size_alignment_validation():
    """GCS chunk size must be a multiple of 256 KiB (262,144 bytes)."""
    valid_size = 512 * 1024  # 512 KiB
    s = Settings(gcs_chunk_size=valid_size)
    assert s.gcs_chunk_size == valid_size

    with pytest.raises(ValidationError) as exc:
        Settings(gcs_chunk_size=1000)
    assert "multiple of 256 KiB" in str(exc.value)


def test_get_settings_caching():
    """Verify get_settings returns a cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
