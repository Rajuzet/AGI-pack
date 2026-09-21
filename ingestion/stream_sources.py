"""Async ingestion client for scientific and technical feeds.

Connects to the ArXiv API and authentic open RSS feeds, normalizes heterogeneous
text (stripping HTML, token limiting), and outputs standardized JSON records
for cloud persistence.
"""

import asyncio
from datetime import datetime, timezone
import html
import json
import logging
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode

from dateutil import parser as date_parser
import feedparser
import httpx
from pydantic import BaseModel, Field

from config import Settings, get_settings

logger = logging.getLogger(__name__)


class StandardizedRecord(BaseModel):
    """Standardized record schema for autonomous agent consumption."""

    id: str = Field(..., description="Unique record identifier (namespaced).")
    source: str = Field(..., description="Ingestion source name (e.g. arxiv, rss:nature_news).")
    title: str = Field(..., description="Cleaned, normalized title.")
    content: str = Field(..., description="Cleaned, normalized body/abstract text.")
    timestamp: str = Field(..., description="ISO 8601 UTC publication timestamp.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific metadata (authors, categories, pdf_url, raw links).",
    )


class TextNormalizer:
    """Text normalization utilities for stripping markup, cleaning whitespace, and bounding tokens."""

    # Precompiled regex patterns for performance
    _SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
    _TAG_RE = re.compile(r"<[^>]+>")
    _WHITESPACE_RE = re.compile(r"[\r\n\t]+")
    _MULTI_SPACE_RE = re.compile(r" {2,}")

    @classmethod
    def strip_html(cls, raw_text: str) -> str:
        """Remove HTML tags, script/style blocks, and decode HTML entities."""
        if not raw_text:
            return ""
        # Remove script and style elements
        clean = cls._SCRIPT_STYLE_RE.sub(" ", raw_text)
        # Strip all HTML tags
        clean = cls._TAG_RE.sub(" ", clean)
        # Unescape HTML entities (&amp;, &lt;, &gt;, &#39;, &nbsp;, etc.)
        clean = html.unescape(clean)
        return clean

    @classmethod
    def normalize_whitespace(cls, text: str) -> str:
        """Collapse newlines, tabs, and multiple spaces into clean single spaces."""
        if not text:
            return ""
        # Normalize newlines and tabs to single spaces
        text = cls._WHITESPACE_RE.sub(" ", text)
        # Collapse multiple horizontal spaces
        text = cls._MULTI_SPACE_RE.sub(" ", text)
        return text.strip()

    @classmethod
    def truncate_tokens(cls, text: str, max_tokens: int = 4000) -> str:
        """Limit text length to an estimated token ceiling, preserving word boundaries.

        Rules of thumb: ~4 characters or ~0.75 words per token. We use conservative
        character budgeting (~4 chars per token) with word-boundary preservation.
        """
        if not text or max_tokens <= 0:
            return text

        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]
        last_space = truncated.rfind(" ")
        if last_space > int(max_chars * 0.8):
            truncated = truncated[:last_space]

        return truncated.rstrip() + "..."

    @classmethod
    def normalize(cls, raw_text: str, max_tokens: int = 4000) -> str:
        """Full pipeline: strip HTML, normalize whitespace, and truncate to token limit."""
        cleaned = cls.strip_html(raw_text)
        normalized = cls.normalize_whitespace(cleaned)
        return cls.truncate_tokens(normalized, max_tokens=max_tokens)


class StreamIngestionClient:
    """High-throughput async ingestion engine for ArXiv and scientific RSS feeds."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize HTTP client and ingestion settings."""
        self.settings: Settings = settings or get_settings()
        self._custom_client = client
        self.normalizer = TextNormalizer

    def _get_http_client(self) -> httpx.AsyncClient:
        """Create or return an httpx.AsyncClient with production headers and timeouts."""
        if self._custom_client:
            return self._custom_client
        return httpx.AsyncClient(
            headers={
                "User-Agent": self.settings.user_agent,
                "Accept": "application/atom+xml, application/xml, text/xml, */*",
            },
            timeout=httpx.Timeout(self.settings.http_timeout_seconds),
            follow_redirects=True,
        )

    def _parse_timestamp(self, raw_date: Any) -> str:
        """Convert heterogeneous dates (struct_time, RFC-822, ISO) into ISO 8601 UTC string."""
        now_iso = datetime.now(timezone.utc).isoformat()
        if not raw_date:
            return now_iso

        try:
            if isinstance(raw_date, time.struct_time):
                dt = datetime.fromtimestamp(time.mktime(raw_date), tz=timezone.utc)
                return dt.isoformat()

            if isinstance(raw_date, str):
                dt = date_parser.parse(raw_date)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt.isoformat()
        except Exception as exc:
            logger.debug("Failed parsing date '%s', defaulting to current time: %s", raw_date, exc)

        return now_iso

    async def _fetch_single_arxiv_category(
        self,
        client: httpx.AsyncClient,
        category: str,
        limit: int,
        start: int = 0,
    ) -> List[StandardizedRecord]:
        """Fetch papers for a single ArXiv category with resilient retries and rate pacing."""
        params = {
            "search_query": f"cat:{category.strip()}",
            "start": str(start),
            "max_results": str(limit),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        # ArXiv Lucene parser expects unescaped colons in query parameter
        query_str = f"search_query=cat:{category.strip()}&start={start}&max_results={limit}&sortBy=submittedDate&sortOrder=descending"
        url = f"{self.settings.arxiv_api_url}?{query_str}"
        logger.info("Fetching ArXiv category '%s' (start=%d): %s", category, start, url)

        feed_text = ""
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                response = await client.get(url)
                if response.status_code in (406, 429, 503):
                    wait_sec = attempt * 3.0
                    logger.warning(
                        "ArXiv rate limit or soft-block (status %d) for cat '%s' on attempt %d/%d. Backing off for %.1fs...",
                        response.status_code,
                        category,
                        attempt,
                        max_attempts,
                        wait_sec,
                    )
                    await asyncio.sleep(wait_sec)
                    continue

                response.raise_for_status()
                feed_text = response.text
                break
            except httpx.HTTPError as exc:
                if attempt == max_attempts:
                    logger.error("ArXiv category '%s' failed after %d attempts: %s", category, max_attempts, exc)
                    return []
                await asyncio.sleep(attempt * 2.0)

        if not feed_text:
            return []

        parsed = feedparser.parse(feed_text)
        records: List[StandardizedRecord] = []

        for entry in parsed.entries:
            try:
                raw_id = entry.get("id", "")
                arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
                record_id = f"arxiv:{arxiv_id}"

                authors = [
                    a.get("name", "").strip()
                    for a in entry.get("authors", [])
                    if a.get("name")
                ]

                pdf_url = None
                for link in entry.get("links", []):
                    if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                        pdf_url = link.get("href")
                        break
                if not pdf_url and arxiv_id:
                    clean_id = arxiv_id.split("v")[0] if "v" in arxiv_id else arxiv_id
                    pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"

                tags = [
                    t.get("term", "").strip()
                    for t in entry.get("tags", [])
                    if t.get("term")
                ]

                title = self.normalizer.normalize(entry.get("title", ""), max_tokens=100)
                content = self.normalizer.normalize(
                    entry.get("summary", ""),
                    max_tokens=self.settings.max_content_tokens,
                )

                pub_raw = entry.get("published_parsed") or entry.get("published")
                timestamp = self._parse_timestamp(pub_raw)

                metadata: Dict[str, Any] = {
                    "authors": authors,
                    "categories": tags,
                    "primary_category": entry.get("arxiv_primary_category", {}).get("term")
                    or (tags[0] if tags else category),
                    "pdf_url": pdf_url,
                    "abs_url": raw_id,
                    "comment": entry.get("arxiv_comment"),
                    "journal_ref": entry.get("arxiv_journal_ref"),
                    "doi": entry.get("arxiv_doi"),
                }

                records.append(
                    StandardizedRecord(
                        id=record_id,
                        source="arxiv",
                        title=title,
                        content=content,
                        timestamp=timestamp,
                        metadata=metadata,
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed ArXiv entry (%s): %s", entry.get("id"), exc)

        return records

    async def fetch_arxiv(
        self,
        categories: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        start: int = 0,
    ) -> List[StandardizedRecord]:
        """Fetch recent papers from the ArXiv API matching specified categories.

        Queries categories with rate pacing and deduplicates papers across categories.

        Args:
            categories: List of categories (e.g. ['cs.AI', 'cs.LG', 'stat.ML']).
            max_results: Upper limit on total unique papers returned.
            start: Starting pagination index for ArXiv query results.

        Returns:
            List of StandardizedRecord objects.
        """
        cats = categories or self.settings.arxiv_categories
        total_limit = max_results or self.settings.arxiv_max_results
        per_cat_limit = max(1, total_limit // len(cats)) if cats else total_limit

        unique_records: Dict[str, StandardizedRecord] = {}

        async with self._get_http_client() as client:
            for i, cat in enumerate(cats):
                if len(unique_records) >= total_limit:
                    break

                if i > 0 and not self._custom_client:
                    # Respect ArXiv 3-second pacing between requests
                    await asyncio.sleep(2.0)

                cat_records = await self._fetch_single_arxiv_category(
                    client=client,
                    category=cat,
                    limit=per_cat_limit,
                    start=start,
                )

                for rec in cat_records:
                    if rec.id not in unique_records:
                        unique_records[rec.id] = rec
                    if len(unique_records) >= total_limit:
                        break

        result = list(unique_records.values())
        logger.info("Successfully ingested %d unique papers across categories %s (start=%d)", len(result), cats, start)
        return result

    async def fetch_rss_feed(
        self,
        feed_name: str,
        feed_url: str,
        max_entries: Optional[int] = None,
    ) -> List[StandardizedRecord]:
        """Fetch and normalize an authentic RSS/Atom feed.

        Args:
            feed_name: Human-readable key for the feed (e.g. 'nature_news').
            feed_url: Target remote feed URL.
            max_entries: Optional limit on processed entries.

        Returns:
            List of StandardizedRecord objects.
        """
        logger.info("Fetching RSS feed '%s' from %s", feed_name, feed_url)

        async with self._get_http_client() as client:
            response = await client.get(feed_url)
            response.raise_for_status()
            feed_text = response.text

        parsed = feedparser.parse(feed_text)
        records: List[StandardizedRecord] = []
        entries = parsed.entries[:max_entries] if max_entries else parsed.entries

        for entry in entries:
            try:
                # Generate unique namespaced identifier
                entry_link = entry.get("link", "")
                raw_guid = entry.get("id") or entry_link
                record_id = f"rss:{feed_name}:{raw_guid}"

                title = self.normalizer.normalize(entry.get("title", ""), max_tokens=150)

                # Find body text across content, summary, and description fields
                raw_body = ""
                if "content" in entry and entry.content:
                    raw_body = entry.content[0].get("value", "")
                elif "summary" in entry:
                    raw_body = entry.get("summary", "")
                elif "description" in entry:
                    raw_body = entry.get("description", "")

                content = self.normalizer.normalize(
                    raw_body,
                    max_tokens=self.settings.max_content_tokens,
                )

                # Parse timestamp
                pub_raw = (
                    entry.get("published_parsed")
                    or entry.get("updated_parsed")
                    or entry.get("published")
                    or entry.get("updated")
                )
                timestamp = self._parse_timestamp(pub_raw)

                # Authors and tags
                authors = [
                    a.get("name", "").strip()
                    for a in entry.get("authors", [])
                    if a.get("name")
                ]
                if not authors and "author" in entry and entry.author:
                    authors = [entry.author.strip()]

                tags = [
                    t.get("term", "").strip()
                    for t in entry.get("tags", [])
                    if t.get("term")
                ]

                metadata: Dict[str, Any] = {
                    "feed_name": feed_name,
                    "feed_url": feed_url,
                    "link": entry_link,
                    "authors": authors,
                    "categories": tags,
                    "feed_title": parsed.feed.get("title", feed_name),
                }

                records.append(
                    StandardizedRecord(
                        id=record_id,
                        source=f"rss:{feed_name}",
                        title=title,
                        content=content,
                        timestamp=timestamp,
                        metadata=metadata,
                    )
                )
            except Exception as exc:
                logger.warning("Skipping malformed RSS entry in feed '%s': %s", feed_name, exc)

        logger.info("Ingested %d records from feed '%s'", len(records), feed_name)
        return records

    async def fetch_all_rss_feeds(
        self,
        feeds: Optional[Dict[str, str]] = None,
        max_entries_per_feed: Optional[int] = None,
    ) -> List[StandardizedRecord]:
        """Concurrently fetch all configured RSS feeds with isolated error boundaries."""
        target_feeds = feeds or self.settings.rss_feeds
        tasks = [
            self.fetch_rss_feed(name, url, max_entries=max_entries_per_feed)
            for name, url in target_feeds.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        aggregated: List[StandardizedRecord] = []

        for feed_name, res in zip(target_feeds.keys(), results):
            if isinstance(res, BaseException):
                logger.error("Failed to fetch RSS feed '%s': %s", feed_name, res)
            else:
                aggregated.extend(res)

        return aggregated

    async def fetch_all_sources(
        self,
        arxiv_limit: Optional[int] = None,
        rss_limit: Optional[int] = None,
        arxiv_start: int = 0,
    ) -> List[StandardizedRecord]:
        """Concurrently fetch both ArXiv and authentic RSS feeds into a unified list."""
        arxiv_task = self.fetch_arxiv(max_results=arxiv_limit, start=arxiv_start)
        rss_task = self.fetch_all_rss_feeds(max_entries_per_feed=rss_limit)

        arxiv_res, rss_res = await asyncio.gather(
            arxiv_task, rss_task, return_exceptions=True
        )

        all_records: List[StandardizedRecord] = []
        if isinstance(arxiv_res, BaseException):
            logger.error("ArXiv ingestion failed: %s", arxiv_res)
        else:
            all_records.extend(arxiv_res)

        if isinstance(rss_res, BaseException):
            logger.error("RSS ingestion failed: %s", rss_res)
        else:
            all_records.extend(rss_res)

        logger.info("Total ingested records across all sources: %d", len(all_records))
        return all_records

    def save_records_to_jsonl(
        self,
        records: List[StandardizedRecord],
        output_file: Union[str, Path],
    ) -> Path:
        """Serialize standardized records into a newline-delimited JSON (JSONL) file.

        Ideal for chunked streaming to Google Cloud Storage.
        """
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.model_dump_json() + "\n")

        logger.info("Wrote %d records to %s (%d bytes)", len(records), path, path.stat().st_size)
        return path
