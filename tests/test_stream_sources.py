"""Unit tests for text normalization, ArXiv parsing, RSS ingestion, and schema compliance."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from ingestion.stream_sources import (
    StandardizedRecord,
    StreamIngestionClient,
    TextNormalizer,
)

SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <title type="html">ArXiv Query: cat:cs.AI</title>
  <id>http://arxiv.org/api/12345</id>
  <updated>2026-09-19T00:00:00Z</updated>
  <entry>
    <id>http://arxiv.org/abs/2609.12345v1</id>
    <updated>2026-09-19T10:00:00Z</updated>
    <published>2026-09-19T08:30:00Z</published>
    <title>Scaling Autonomous Reasoning: A Survey &amp; Benchmark</title>
    <summary> Recent advances in autonomous agents demonstrated &lt;b&gt;unprecedented&lt;/b&gt; reasoning capability.
    We introduce a novel framework for resilient cloud orchestration. </summary>
    <author><name>Alice Turing</name></author>
    <author><name>Bob Shannon</name></author>
    <arxiv:doi>10.1000/182</arxiv:doi>
    <arxiv:comment>Submitted to NeurIPS 2026</arxiv:comment>
    <arxiv:primary_category term="cs.AI"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2609.12345v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2609.12345v1" rel="related" type="application/pdf"/>
  </entry>
</feed>"""

SAMPLE_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Nature News - Latest Science</title>
    <link>https://www.nature.com</link>
    <description>Daily scientific reporting</description>
    <item>
      <title>Breakthrough in Quantum Computing &amp; Machine Learning</title>
      <link>https://www.nature.com/articles/d41586-026-0001</link>
      <guid isPermaLink="true">https://www.nature.com/articles/d41586-026-0001</guid>
      <pubDate>Fri, 19 Sep 2026 12:00:00 +0000</pubDate>
      <author>dr.science@nature.com (Jane Doe)</author>
      <category>Quantum Physics</category>
      <description>&lt;p&gt;Researchers have demonstrated fault-tolerant quantum algorithms &lt;style&gt;p { color: red; }&lt;/style&gt; coupled with neural nets.&lt;/p&gt;</description>
    </item>
  </channel>
</rss>"""


def test_text_normalizer_strip_html():
    """Verify HTML stripping, style/script tag removal, and entity unescaping."""
    raw = """
    <div>
      <script>console.log('remove me');</script>
      <h1>AI Breakthrough &amp; Future</h1>
      <p>Researchers observed <b>&lt;significant&gt;</b> gains &#39;in-the-wild&#39;.</p>
    </div>
    """
    cleaned = TextNormalizer.strip_html(raw)
    assert "<script>" not in cleaned
    assert "console.log" not in cleaned
    assert "<b>" not in cleaned
    assert "AI Breakthrough & Future" in cleaned
    assert "<significant>" in cleaned
    assert "'in-the-wild'" in cleaned


def test_text_normalizer_whitespace():
    """Verify tabs, newlines, and multiple spaces are collapsed."""
    raw = "  Hello \t\t world \n\n this   is \r\n a   test.   "
    normalized = TextNormalizer.normalize_whitespace(raw)
    assert normalized == "Hello world this is a test."


def test_text_normalizer_token_truncation():
    """Verify text is cleanly truncated at word boundary when exceeding token limit."""
    long_text = "Word " * 200  # 1000 characters
    # Max tokens = 20 (~80 chars)
    truncated = TextNormalizer.truncate_tokens(long_text, max_tokens=20)
    assert len(truncated) <= 20 * 4 + 3
    assert truncated.endswith("...")
    # Word shouldn't be sliced in half
    assert not truncated.endswith("Wor...")


def test_standardized_record_schema():
    """Verify record schema validation enforces required fields."""
    rec = StandardizedRecord(
        id="arxiv:2609.12345v1",
        source="arxiv",
        title="Valid Title",
        content="Abstract content",
        timestamp="2026-09-19T08:30:00Z",
        metadata={"authors": ["Alice Turing"], "pdf_url": "https://arxiv.org/pdf/2609.12345v1.pdf"},
    )
    d = rec.model_dump()
    assert d["id"] == "arxiv:2609.12345v1"
    assert d["source"] == "arxiv"
    assert d["title"] == "Valid Title"
    assert d["metadata"]["pdf_url"] == "https://arxiv.org/pdf/2609.12345v1.pdf"


@pytest.mark.asyncio
async def test_fetch_arxiv_parsing():
    """Verify ArXiv API response parsing into StandardizedRecord objects."""
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.text = SAMPLE_ARXIV_XML
    mock_resp.raise_for_status = MagicMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get.return_value = mock_resp

    client = StreamIngestionClient(client=mock_client)
    records = await client.fetch_arxiv(categories=["cs.AI"], max_results=1)

    assert len(records) == 1
    record = records[0]
    assert record.id == "arxiv:2609.12345v1"
    assert record.source == "arxiv"
    assert record.title == "Scaling Autonomous Reasoning: A Survey & Benchmark"
    assert "Recent advances in autonomous agents demonstrated" in record.content
    assert "<b>" not in record.content
    assert record.metadata["authors"] == ["Alice Turing", "Bob Shannon"]
    assert record.metadata["pdf_url"] == "http://arxiv.org/pdf/2609.12345v1"
    assert "cs.AI" in record.metadata["categories"]
    assert record.metadata["doi"] == "10.1000/182"
    assert "2026-09-19" in record.timestamp


@pytest.mark.asyncio
async def test_fetch_rss_feed_parsing():
    """Verify RSS 2.0 feed parsing into StandardizedRecord objects."""
    mock_client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.text = SAMPLE_RSS_XML
    mock_resp.raise_for_status = MagicMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.get.return_value = mock_resp

    client = StreamIngestionClient(client=mock_client)
    records = await client.fetch_rss_feed(
        feed_name="nature_news",
        feed_url="https://www.nature.com/nature.rss",
    )

    assert len(records) == 1
    record = records[0]
    assert record.source == "rss:nature_news"
    assert "Breakthrough in Quantum Computing & Machine Learning" in record.title
    assert "Researchers have demonstrated fault-tolerant quantum algorithms" in record.content
    assert "<style>" not in record.content
    assert record.metadata["link"] == "https://www.nature.com/articles/d41586-026-0001"
    assert "2026-09-19" in record.timestamp


def test_save_records_to_jsonl(tmp_path):
    """Verify staging records to valid JSONL output."""
    client = StreamIngestionClient()
    records = [
        StandardizedRecord(
            id=f"test:{i}",
            source="test",
            title=f"Title {i}",
            content=f"Content {i}",
            timestamp="2026-09-19T00:00:00Z",
            metadata={"index": i},
        )
        for i in range(3)
    ]

    out_file = tmp_path / "staged.jsonl"
    saved = client.save_records_to_jsonl(records, out_file)

    assert saved.is_file()
    lines = saved.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3
    loaded = [json.loads(line) for line in lines]
    assert loaded[0]["id"] == "test:0"
    assert loaded[2]["metadata"]["index"] == 2
