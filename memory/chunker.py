"""Semantic text chunker with sentence boundary detection and token window overlap.

Splits technical documents, preprints, and articles into coherent semantic chunks
optimized for dense vector embedding and high-precision information retrieval.
"""

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SemanticChunker:
    """Splits documents into semantic passages bounded by token windows and sentence limits."""

    # Common English abbreviations and honorifics that should not trigger sentence boundaries
    _ABBREVIATIONS = (
        r"\b(?:e\.g|i\.e|et al|vs|etc|Dr|Mr|Mrs|Ms|Prof|Fig|no|vol|dept|univ|approx|est)\."
    )
    _ABBR_RE = re.compile(_ABBREVIATIONS, re.IGNORECASE)

    # Sentence boundary: punctuation (. ! ?) followed by whitespace and an uppercase letter or quote
    _SENTENCE_SPLIT_RE = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(])",
        re.UNICODE,
    )

    def __init__(
        self,
        default_max_tokens: int = 512,
        default_overlap_tokens: int = 50,
        chars_per_token: int = 4,
    ) -> None:
        """Initialize chunker with token bounds and character estimation ratio."""
        if default_overlap_tokens >= default_max_tokens:
            raise ValueError("default_overlap_tokens must be strictly less than default_max_tokens.")

        self.default_max_tokens = default_max_tokens
        self.default_overlap_tokens = default_overlap_tokens
        self.chars_per_token = chars_per_token

    def _protect_abbreviations(self, text: str) -> str:
        """Replace periods in known abbreviations with a sentinel to prevent false splits."""
        def repl(match: re.Match) -> str:
            return match.group(0).replace(".", "§DOT§")
        return self._ABBR_RE.sub(repl, text)

    def _unprotect_abbreviations(self, text: str) -> str:
        """Restore sentinel placeholders back to periods."""
        return text.replace("§DOT§", ".")

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences while respecting abbreviations and numeric decimals."""
        if not text or not text.strip():
            return []

        clean_text = text.strip()
        # Protect abbreviations
        protected = self._protect_abbreviations(clean_text)
        # Split on sentence boundaries
        raw_sentences = self._SENTENCE_SPLIT_RE.split(protected)
        # Unprotect and strip each sentence
        sentences = [
            self._unprotect_abbreviations(s).strip()
            for s in raw_sentences
            if s and s.strip()
        ]
        return sentences

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count based on character length (~4 chars/token)."""
        return max(1, len(text) // self.chars_per_token)

    def _split_long_sentence(self, sentence: str, max_chars: int, overlap_chars: int) -> List[str]:
        """Sub-split an oversized single sentence on word boundaries."""
        words = sentence.split()
        chunks: List[str] = []
        current_words: List[str] = []
        current_len = 0

        for word in words:
            word_len = len(word) + 1
            if current_words and (current_len + word_len) > max_chars:
                chunk_str = " ".join(current_words).strip()
                if chunk_str:
                    chunks.append(chunk_str)

                # Keep trailing words for overlap
                overlap_words: List[str] = []
                overlap_len = 0
                for w in reversed(current_words):
                    if overlap_len + len(w) + 1 <= overlap_chars:
                        overlap_words.insert(0, w)
                        overlap_len += len(w) + 1
                    else:
                        break
                current_words = overlap_words
                current_len = sum(len(w) + 1 for w in current_words)

            current_words.append(word)
            current_len += word_len

        if current_words:
            chunk_str = " ".join(current_words).strip()
            if chunk_str:
                chunks.append(chunk_str)

        return chunks

    def split_text(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
    ) -> List[str]:
        """Split arbitrary text into overlapping chunks respecting sentence boundaries.

        Args:
            text: Input string to chunk.
            max_tokens: Maximum tokens per chunk (defaults to configured 512).
            overlap_tokens: Token overlap between consecutive chunks (defaults to 50).

        Returns:
            List of chunk strings.
        """
        if not text or not text.strip():
            return []

        limit_tokens = max_tokens or self.default_max_tokens
        overlap = overlap_tokens if overlap_tokens is not None else self.default_overlap_tokens

        if overlap >= limit_tokens:
            raise ValueError(f"overlap_tokens ({overlap}) must be less than max_tokens ({limit_tokens}).")

        max_chars = limit_tokens * self.chars_per_token
        overlap_chars = overlap * self.chars_per_token

        sentences = self.split_sentences(text)
        if not sentences:
            return []

        # Flatten sentences that individually exceed max_chars
        normalized_units: List[str] = []
        for s in sentences:
            if len(s) > max_chars:
                normalized_units.extend(self._split_long_sentence(s, max_chars, overlap_chars))
            else:
                normalized_units.append(s)

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_chars = 0

        for unit in normalized_units:
            unit_chars = len(unit) + 1  # include space

            if current_sentences and (current_chars + unit_chars) > max_chars:
                # Emit current chunk
                chunk_str = " ".join(current_sentences).strip()
                if chunk_str:
                    chunks.append(chunk_str)

                # Compute sentence overlap for next chunk
                overlap_sentences: List[str] = []
                acc_overlap_chars = 0
                for s in reversed(current_sentences):
                    if acc_overlap_chars + len(s) + 1 <= overlap_chars:
                        overlap_sentences.insert(0, s)
                        acc_overlap_chars += len(s) + 1
                    else:
                        break

                # If no full sentence fits in overlap_chars, take trailing words from the last sentence
                if not overlap_sentences and overlap_chars > 0 and current_sentences:
                    last_s = current_sentences[-1]
                    words = last_s.split()
                    sub_words: List[str] = []
                    sub_len = 0
                    for w in reversed(words):
                        if sub_len + len(w) + 1 <= overlap_chars:
                            sub_words.insert(0, w)
                            sub_len += len(w) + 1
                        else:
                            break
                    if sub_words:
                        overlap_sentences = [" ".join(sub_words)]

                current_sentences = overlap_sentences
                current_chars = sum(len(s) + 1 for s in current_sentences)

            current_sentences.append(unit)
            current_chars += unit_chars

        if current_sentences:
            chunk_str = " ".join(current_sentences).strip()
            if chunk_str:
                chunks.append(chunk_str)

        return chunks

    def chunk_document(
        self,
        doc: Dict[str, Any],
        max_tokens: Optional[int] = None,
        overlap_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk a standardized record or document dictionary and preserve provenance.

        Args:
            doc: Input document containing 'id', 'title', 'content' (or 'text'), and 'metadata'.
            max_tokens: Maximum tokens per chunk.
            overlap_tokens: Token overlap.

        Returns:
            List of chunk records with unique IDs and enriched metadata.
        """
        doc_id = str(doc.get("id", "doc_unknown"))
        title = doc.get("title", "")
        content = doc.get("content") or doc.get("text") or ""
        source = doc.get("source", "unknown")
        timestamp = doc.get("timestamp", "")
        base_meta = doc.get("metadata", {})

        # If title is present, prepend to first chunk or chunk text for richer context
        full_text = f"{title}\n\n{content}" if title and content else (content or title)
        chunks = self.split_text(full_text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

        if not chunks:
            # Fallback for empty content
            return []

        total_chunks = len(chunks)
        chunk_records: List[Dict[str, Any]] = []

        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{idx}"
            chunk_meta = {
                **base_meta,
                "parent_id": doc_id,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "estimated_tokens": self._estimate_tokens(chunk_text),
            }

            chunk_records.append(
                {
                    "id": chunk_id,
                    "parent_id": doc_id,
                    "chunk_index": idx,
                    "total_chunks": total_chunks,
                    "text": chunk_text,
                    "title": title,
                    "source": source,
                    "timestamp": timestamp,
                    "metadata": chunk_meta,
                }
            )

        return chunk_records
