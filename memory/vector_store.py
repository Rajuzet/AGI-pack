"""High-efficiency vector memory engine powered by FAISS and SentenceTransformers.

Supports local CPU/GPU dense embeddings, cosine similarity via L2-normalized IndexFlatIP,
thread-safe atomic updates, local persistence, and bidirectional Google Cloud Storage synchronization.
"""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import faiss
import numpy as np

from config import Settings, get_settings
from memory.chunker import SemanticChunker
from storage.gcs_manager import GCSManager

logger = logging.getLogger(__name__)


class VectorStore:
    """Thread-safe, memory-bounded vector store utilizing FAISS and SentenceTransformers.

    Features:
        - Dense vector generation using local SentenceTransformers (all-MiniLM-L6-v2 by default).
        - Cosine similarity search via FAISS IndexFlatIP on L2-normalized vectors.
        - Integrated semantic chunking for long-form records.
        - Atomic local disk persistence (`index.faiss` and `metadata.json`).
        - Bidirectional Google Cloud Storage synchronization with cold-start tolerance.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        embedder: Optional[Any] = None,
        local_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the vector store with FAISS index and embedding engine."""
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.embedding_model_name
        self.dimension: int = int(dimension or self.settings.vector_dimension)
        self.local_dir = Path(local_dir or self.settings.memory_local_dir)

        # Thread synchronization lock
        self._lock = threading.RLock()

        # Semantic chunker for unsegmented inputs
        self.chunker = SemanticChunker(
            default_max_tokens=self.settings.chunk_max_tokens,
            default_overlap_tokens=self.settings.chunk_overlap_tokens,
        )

        # Embedding model (lazy or injected)
        self._embedder = embedder

        # Initialize empty FAISS IndexFlatIP (Inner Product)
        self.index: faiss.Index = faiss.IndexFlatIP(self.dimension)

        # Internal storage for document metadata mapped by vector index (0 .. N-1)
        self.metadata_map: Dict[int, Dict[str, Any]] = {}
        self.doc_ids: Set[str] = set()

        logger.info(
            "VectorStore initialized (model=%s, dim=%d, metric=Cosine/InnerProduct)",
            self.model_name,
            self.dimension,
        )

    @property
    def embedding_dimension(self) -> int:
        """Return the vector embedding dimension."""
        return self.dimension

    @property
    def model(self) -> Any:
        """Alias for embedder SentenceTransformer model."""
        return self.embedder

    @property
    def embedder(self) -> Any:
        """Lazy-initialize and return the SentenceTransformer embedding model."""
        if self._embedder is not None:
            return self._embedder

        with self._lock:
            if self._embedder is not None:
                return self._embedder

            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Loading SentenceTransformer model '%s' on device '%s'...", self.model_name, device)
            self._embedder = SentenceTransformer(self.model_name, device=device)
            # Validate model dimension matches configured dimension
            if hasattr(self._embedder, "get_embedding_dimension"):
                test_dim = self._embedder.get_embedding_dimension()
            else:
                test_dim = self._embedder.get_sentence_embedding_dimension()
            if test_dim is not None and test_dim != self.dimension:
                logger.warning(
                    "Configured vector dimension (%d) does not match model embedding dimension (%d). Updating to %d.",
                    self.dimension,
                    test_dim,
                    test_dim,
                )
                self.dimension = int(test_dim)
                self.index = faiss.IndexFlatIP(self.dimension)

            return self._embedder

    def _encode_texts(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode a batch of texts into L2-normalized float32 numpy vectors."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # SentenceTransformer handles normalization internally when requested
        embeddings = self.embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings_np = np.asarray(embeddings, dtype=np.float32)
        # Ensure L2 normalization for exact cosine similarity under IndexFlatIP
        faiss.normalize_L2(embeddings_np)
        return embeddings_np

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_documents: bool = True,
        batch_size: int = 64,
    ) -> List[str]:
        """Embed text passages, append vectors to FAISS index, and record metadata.

        Args:
            documents: List of document dicts (StandardizedRecord or chunk dicts).
            chunk_documents: If True, segments large document texts using SemanticChunker.
            batch_size: Batch size for dense embedding computation.

        Returns:
            List of generated/appended chunk record IDs.
        """
        if not documents:
            return []

        # Prepare passages
        passages: List[Dict[str, Any]] = []
        for doc in documents:
            if chunk_documents and (doc.get("content") or len(doc.get("text", "")) > 1000):
                chunks = self.chunker.chunk_document(doc)
                passages.extend(chunks)
            else:
                doc_id = str(doc.get("id") or f"doc_{len(self.doc_ids)}")
                text = doc.get("text") or doc.get("content") or doc.get("title") or ""
                passages.append(
                    {
                        "id": doc_id,
                        "text": text,
                        "title": doc.get("title", ""),
                        "source": doc.get("source", "unknown"),
                        "timestamp": doc.get("timestamp", ""),
                        "metadata": doc.get("metadata", {}),
                    }
                )

        if not passages:
            return []

        with self._lock:
            # Filter out duplicates if already present
            new_passages: List[Dict[str, Any]] = []
            for p in passages:
                pid = p["id"]
                if pid not in self.doc_ids:
                    new_passages.append(p)
                    self.doc_ids.add(pid)

            if not new_passages:
                logger.info("All %d incoming document passages are already indexed.", len(passages))
                return []

            texts_to_embed = [p["text"] for p in new_passages]
            vectors = self._encode_texts(texts_to_embed, batch_size=batch_size)

            start_idx = self.index.ntotal
            self.index.add(vectors)

            added_ids: List[str] = []
            for i, passage in enumerate(new_passages):
                vector_idx = start_idx + i
                self.metadata_map[vector_idx] = passage
                added_ids.append(passage["id"])

            logger.info(
                "Added %d vectors to FAISS index (total vectors: %d)",
                len(added_ids),
                self.index.ntotal,
            )
            return added_ids

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search for a query string.

        Args:
            query: Natural language query.
            top_k: Number of most relevant nearest neighbors to return.
            score_threshold: Optional minimum cosine similarity score cutoff (-1.0 to 1.0).

        Returns:
            List of matching records ordered by similarity score descending.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        with self._lock:
            total_items = self.index.ntotal
            if total_items == 0:
                logger.debug("VectorStore index is empty; 0 results returned.")
                return []

            k = min(top_k, total_items)
            query_vector = self._encode_texts([clean_query], batch_size=1)

            scores, indices = self.index.search(query_vector, k)

            results: List[Dict[str, Any]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue  # FAISS padding sentinel

                sim_score = float(score)
                if score_threshold is not None and sim_score < score_threshold:
                    continue

                meta = self.metadata_map.get(int(idx), {})
                results.append(
                    {
                        "id": meta.get("id"),
                        "score": round(sim_score, 4),
                        "text": meta.get("text", ""),
                        "title": meta.get("title", ""),
                        "source": meta.get("source", ""),
                        "timestamp": meta.get("timestamp", ""),
                        "metadata": meta.get("metadata", {}),
                    }
                )

            return results

    def save_local(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Persist the FAISS index and metadata map locally to disk.

        Args:
            path: Target directory (defaults to self.local_dir).

        Returns:
            Resolved Path to the storage directory.
        """
        target_dir = Path(path or self.local_dir).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / "index.faiss"
        metadata_file = target_dir / "metadata.json"

        with self._lock:
            # Write FAISS index
            faiss.write_index(self.index, str(index_file))

            # Write metadata catalog
            catalog = {
                "version": "1.0",
                "model_name": self.model_name,
                "dimension": self.dimension,
                "total_vectors": self.index.ntotal,
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "metadata_map": {str(k): v for k, v in self.metadata_map.items()},
            }

            temp_meta = metadata_file.with_suffix(".json.tmp")
            with temp_meta.open("w", encoding="utf-8") as f:
                json.dump(catalog, f, ensure_ascii=False, indent=2)
            temp_meta.replace(metadata_file)

            logger.info(
                "Persisted vector memory to '%s' (%d vectors, index=%d bytes)",
                target_dir,
                self.index.ntotal,
                index_file.stat().st_size,
            )
            return target_dir

    def load_local(self, path: Optional[Union[str, Path]] = None) -> bool:
        """Load an existing FAISS index and metadata catalog from disk.

        Args:
            path: Source directory (defaults to self.local_dir).

        Returns:
            True if loaded successfully, False if files do not exist.
        """
        source_dir = Path(path or self.local_dir).resolve()
        index_file = source_dir / "index.faiss"
        metadata_file = source_dir / "metadata.json"

        if not index_file.is_file() or not metadata_file.is_file():
            logger.warning("Local vector memory files missing at: %s", source_dir)
            return False

        with self._lock:
            try:
                loaded_index = faiss.read_index(str(index_file))
                with metadata_file.open("r", encoding="utf-8") as f:
                    catalog = json.load(f)

                # Reconstruct metadata map
                raw_map = catalog.get("metadata_map", {})
                new_metadata_map = {int(k): v for k, v in raw_map.items()}
                new_doc_ids = {v["id"] for v in new_metadata_map.values() if "id" in v}

                self.index = loaded_index
                self.dimension = loaded_index.d
                self.metadata_map = new_metadata_map
                self.doc_ids = new_doc_ids

                logger.info(
                    "Successfully loaded vector memory from '%s' (%d vectors)",
                    source_dir,
                    self.index.ntotal,
                )
                return True
            except Exception as exc:
                logger.error("Failed loading local vector memory from '%s': %s", source_dir, exc)
                return False

    def sync_to_gcs(
        self,
        gcs_manager: GCSManager,
        remote_folder: Optional[str] = None,
        local_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Flush the local FAISS index and metadata catalog directly to Google Cloud Storage.

        Args:
            gcs_manager: Configured GCSManager instance.
            remote_folder: Destination GCS folder prefix (defaults to config).
            local_path: Local directory to save before upload.

        Returns:
            Dictionary with upload statuses and remote cloud URIs.
        """
        target_prefix = (remote_folder or self.settings.memory_remote_prefix).strip("/")
        saved_dir = self.save_local(local_path)

        index_local = saved_dir / "index.faiss"
        meta_local = saved_dir / "metadata.json"

        remote_index = f"{target_prefix}/index.faiss"
        remote_meta = f"{target_prefix}/metadata.json"

        logger.info(
            "Syncing vector memory to GCS: %s -> gs://%s/%s",
            saved_dir,
            gcs_manager.bucket_name,
            target_prefix,
        )

        try:
            index_uri = gcs_manager.upload_file(
                index_local,
                remote_index,
                content_type="application/octet-stream",
            )
            meta_uri = gcs_manager.upload_file(
                meta_local,
                remote_meta,
                content_type="application/json",
            )

            summary = {
                "status": "synced",
                "remote_folder": target_prefix,
                "total_vectors": self.index.ntotal,
                "index_uri": index_uri,
                "metadata_uri": meta_uri,
                "synced_at": datetime.now(timezone.utc).isoformat(),
            }
            logger.info("Successfully synced vector memory to GCS: %s", summary)
            return summary
        except Exception as exc:
            logger.warning("Failed to sync vector memory to GCS: %s. Operating in local-only mode.", exc)
            return {
                "status": "failed",
                "error": str(exc),
                "remote_folder": target_prefix,
                "total_vectors": self.index.ntotal,
                "local_dir": str(saved_dir),
            }

    def sync_from_gcs(
        self,
        gcs_manager: GCSManager,
        remote_folder: Optional[str] = None,
        local_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """Download remote FAISS index from GCS if available, or gracefully handle cold start.

        Args:
            gcs_manager: Configured GCSManager instance.
            remote_folder: Target GCS folder prefix.
            local_path: Local destination directory for index files.

        Returns:
            True if remote index was loaded, False if cold-start (new index initialized).
        """
        target_prefix = (remote_folder or self.settings.memory_remote_prefix).strip("/")
        dest_dir = Path(local_path or self.local_dir).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)

        remote_index = f"{target_prefix}/index.faiss"
        remote_meta = f"{target_prefix}/metadata.json"

        logger.info(
            "Checking remote vector memory at gs://%s/%s...",
            gcs_manager.bucket_name,
            target_prefix,
        )

        try:
            # Check existence of remote blobs
            index_blob = gcs_manager.bucket.blob(remote_index)
            meta_blob = gcs_manager.bucket.blob(remote_meta)

            if not index_blob.exists(gcs_manager.client) or not meta_blob.exists(gcs_manager.client):
                logger.info(
                    "No existing vector memory found at gs://%s/%s. Starting cold with an empty index.",
                    gcs_manager.bucket_name,
                    target_prefix,
                )
                return False

            # Download chunked streams
            gcs_manager.download_file(remote_index, dest_dir / "index.faiss")
            gcs_manager.download_file(remote_meta, dest_dir / "metadata.json")

            # Load into active FAISS index
            return self.load_local(dest_dir)
        except Exception as exc:
            logger.warning(
                "Unable to pull vector memory from GCS (%s). Initializing cold empty index.",
                exc,
            )
            return False
