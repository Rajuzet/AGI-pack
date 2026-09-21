"""Unit tests for semantic chunker and text splitting."""

import pytest

from memory.chunker import SemanticChunker


def test_sentence_splitting_with_abbreviations():
    """Ensure sentence splitting protects abbreviations and decimal numbers."""
    chunker = SemanticChunker()
    text = (
        "Dr. Smith presented findings (e.g. at NeurIPS 2026). "
        "The model achieved 98.5% accuracy et al. verified this result. "
        "Furthermore, see Fig. 4 for structural architecture."
    )
    sentences = chunker.split_sentences(text)
    assert len(sentences) == 3
    assert "Dr. Smith" in sentences[0]
    assert "(e.g. at NeurIPS 2026)." in sentences[0]
    assert "98.5% accuracy" in sentences[1]
    assert "et al. verified this result." in sentences[1]
    assert "Fig. 4" in sentences[2]


def test_chunking_token_window_and_overlap():
    """Verify max_tokens bound and overlap preservation between adjacent chunks."""
    chunker = SemanticChunker(default_max_tokens=30, default_overlap_tokens=10)
    # ~30 tokens is ~120 characters
    sentences = [
        "First sentence discussing autonomous agent planning and reasoning.",
        "Second sentence covering reinforcement learning with environment rewards.",
        "Third sentence analyzing dense vector representations and cosine metric.",
        "Fourth sentence evaluating zero-shot generalization on benchmarks.",
    ]
    full_text = " ".join(sentences)
    chunks = chunker.split_text(full_text, max_tokens=30, overlap_tokens=10)

    assert len(chunks) >= 2
    for chunk in chunks:
        # Estimated tokens should not drastically exceed max_tokens
        assert chunker._estimate_tokens(chunk) <= 35

    # Check overlap: second chunk should share text with first chunk
    assert any(w in chunks[1] for w in chunks[0].split()[-5:])


def test_chunk_document_metadata_preservation():
    """Verify document metadata, parent_id, and chunk_index are properly tracked."""
    chunker = SemanticChunker(default_max_tokens=25, default_overlap_tokens=5)
    doc = {
        "id": "arxiv:2609.99999",
        "title": "Scaling Laws in Autonomous Intelligence",
        "content": (
            "Recent architectures demonstrate strong reasoning capabilities. "
            "We conduct extensive empirical evaluations across diverse tasks. "
            "Our results demonstrate that memory consolidation yields persistent gains. "
            "Future work will explore hierarchical multi-agent coordination."
        ),
        "source": "arxiv",
        "timestamp": "2026-09-19T12:00:00Z",
        "metadata": {"authors": ["Alice Turing"], "pdf_url": "https://arxiv.org/pdf/test.pdf"},
    }

    chunks = chunker.chunk_document(doc, max_tokens=25, overlap_tokens=5)
    assert len(chunks) >= 2

    for idx, c in enumerate(chunks):
        assert c["parent_id"] == "arxiv:2609.99999"
        assert c["chunk_index"] == idx
        assert c["total_chunks"] == len(chunks)
        assert c["id"] == f"arxiv:2609.99999_chunk_{idx}"
        assert c["title"] == "Scaling Laws in Autonomous Intelligence"
        assert c["source"] == "arxiv"
        assert c["metadata"]["authors"] == ["Alice Turing"]
        assert c["metadata"]["pdf_url"] == "https://arxiv.org/pdf/test.pdf"


def test_invalid_overlap_raises_error():
    """Ensure invalid overlap >= max_tokens raises ValueError."""
    chunker = SemanticChunker()
    with pytest.raises(ValueError):
        chunker.split_text("Sample text", max_tokens=20, overlap_tokens=25)
