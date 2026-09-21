"""Unit tests for VectorStore indexing, similarity search, persistence, and GCS sync."""

import json
from pathlib import Path
import threading
from unittest.mock import MagicMock
import numpy as np
import pytest

from config import Settings
from memory.vector_store import VectorStore


class MockEmbedder:
    """Fast, deterministic mock embedder generating normalized vectors for testing."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ):
        import hashlib
        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            words = text.lower().split()
            for w in words:
                h_val = int(hashlib.sha256(w.encode("utf-8")).hexdigest()[:8], 16)
                idx = h_val % self.dimension
                vec[idx] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            else:
                vec[0] = 1.0
            vectors.append(vec)
        return np.vstack(vectors)


@pytest.fixture
def mock_vector_store(tmp_path):
    """Provide an isolated VectorStore with a mock embedder and temp directory."""
    embedder = MockEmbedder(dimension=64)
    settings = Settings(
        vector_dimension=64,
        memory_local_dir=tmp_path / "memory_store",
    )
    vs = VectorStore(settings=settings, dimension=64, embedder=embedder, local_dir=tmp_path / "memory_store")
    return vs


def test_add_documents_and_search(mock_vector_store):
    """Verify adding documents increases index size and allows similarity search."""
    docs = [
        {
            "id": "doc_1",
            "title": "Quantum Computing",
            "content": "Superconducting qubits enable quantum algorithms with exponential speedup.",
            "source": "nature_news",
            "metadata": {"category": "Physics"},
        },
        {
            "id": "doc_2",
            "title": "Autonomous Agents",
            "content": "Large language models serve as central reasoning planners for robotics.",
            "source": "arxiv",
            "metadata": {"category": "AI"},
        },
    ]

    added = mock_vector_store.add_documents(docs, chunk_documents=False)
    assert len(added) == 2
    assert mock_vector_store.index.ntotal == 2

    # Query matching doc_2
    results = mock_vector_store.similarity_search("Autonomous Agents and language models", top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "doc_2"
    assert results[0]["title"] == "Autonomous Agents"
    assert -1.0 <= results[0]["score"] <= 1.01


def test_save_and_load_local(mock_vector_store, tmp_path):
    """Verify local disk persistence round-trip."""
    docs = [
        {"id": "doc_a", "content": "Alpha passage text", "source": "src1"},
        {"id": "doc_b", "content": "Beta passage text", "source": "src2"},
    ]
    mock_vector_store.add_documents(docs, chunk_documents=False)
    assert mock_vector_store.index.ntotal == 2

    save_dir = tmp_path / "persisted_store"
    mock_vector_store.save_local(save_dir)

    assert (save_dir / "index.faiss").is_file()
    assert (save_dir / "metadata.json").is_file()

    # Create fresh VectorStore and load from disk
    new_vs = VectorStore(
        dimension=16,
        embedder=MockEmbedder(dimension=16),
        local_dir=save_dir,
    )
    success = new_vs.load_local(save_dir)
    assert success is True
    assert new_vs.index.ntotal == 2
    assert 0 in new_vs.metadata_map
    assert new_vs.metadata_map[0]["id"] == "doc_a"


def test_thread_safety_concurrent_indexing(mock_vector_store):
    """Verify thread-safe additions under concurrent execution."""
    threads = []
    errors = []

    def worker(worker_id: int):
        try:
            docs = [
                {
                    "id": f"thread_{worker_id}_doc_{i}",
                    "content": f"Worker {worker_id} passage content {i}",
                }
                for i in range(5)
            ]
            mock_vector_store.add_documents(docs, chunk_documents=False)
        except Exception as exc:
            errors.append(exc)

    for t_id in range(4):
        t = threading.Thread(target=worker, args=(t_id,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    assert len(errors) == 0
    assert mock_vector_store.index.ntotal == 20


def test_sync_to_and_from_gcs(mock_vector_store, tmp_path):
    """Verify bidirectional GCS synchronization using mock GCSManager."""
    docs = [{"id": "cloud_doc_1", "content": "Distributed cloud memory item"}]
    mock_vector_store.add_documents(docs, chunk_documents=False)

    mock_gcs = MagicMock()
    mock_gcs.bucket_name = "test-agi-bucket"
    mock_gcs.upload_file.side_effect = lambda local, remote, **kw: f"gs://test-agi-bucket/{remote}"

    # 1. Test sync_to_gcs
    sync_res = mock_vector_store.sync_to_gcs(mock_gcs, remote_folder="vector_memory/prod")
    assert sync_res["status"] == "synced"
    assert sync_res["total_vectors"] == 1
    assert "vector_memory/prod/index.faiss" in sync_res["index_uri"]

    # 2. Test sync_from_gcs cold start (remote blobs do not exist)
    mock_blob = MagicMock()
    mock_blob.exists.return_value = False
    mock_gcs.bucket.blob.return_value = mock_blob

    fresh_vs = VectorStore(dimension=16, embedder=MockEmbedder(dimension=16), local_dir=tmp_path / "cold_local")
    cold_loaded = fresh_vs.sync_from_gcs(mock_gcs, remote_folder="vector_memory/empty")
    assert cold_loaded is False  # Handled cold-start gracefully
    assert fresh_vs.index.ntotal == 0
