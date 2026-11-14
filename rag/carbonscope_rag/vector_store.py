"""Step 4: store and search vectors in ChromaDB."""
import logging
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import numpy as np
from langchain_core.documents import Document

from . import config

logger = logging.getLogger(__name__)

_BATCH = 500  # keep each Chroma write comfortably under its batch limit


class VectorStore:
    def __init__(
        self,
        persist_directory: Path = config.VECTOR_DIR,
        collection_name: str = config.COLLECTION_NAME,
    ):
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))
        self.collection = self._get_collection()

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete everything and start with an empty collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self._get_collection()
        logger.info("Collection '%s' reset", self.collection_name)

    def delete_source(self, source_file: str) -> None:
        """Remove all chunks of one PDF (used before re-ingesting it, so edited
        PDFs don't leave stale chunks behind)."""
        self.collection.delete(where={"source_file": source_file})

    def upsert(self, chunks: List[Document], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        for start in range(0, len(chunks), _BATCH):
            batch = chunks[start : start + _BATCH]
            batch_emb = embeddings[start : start + _BATCH]
            self.collection.upsert(
                ids=[c.metadata["chunk_id"] for c in batch],
                documents=[c.page_content for c in batch],
                embeddings=[e.tolist() for e in batch_emb],
                # Chroma only accepts simple values, so keep just the fields we use
                metadatas=[
                    {
                        "source_file": c.metadata["source_file"],
                        "page": c.metadata["page"],
                        "total_pages": c.metadata.get("total_pages", 0),
                        "chunk_index": c.metadata["chunk_index"],
                    }
                    for c in batch
                ],
            )
        logger.info("Upserted %d chunks. Collection now holds %d.", len(chunks), self.count())

    def query(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Return the top_k nearest chunks, best first, each with a cosine 'score'."""
        total = self.count()
        if total == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append({"content": doc, "metadata": meta, "score": 1.0 - dist})
        return hits
