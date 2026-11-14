"""Step 5: find the chunks relevant to a question."""
import logging
from typing import Any, Dict, List

from . import config
from .embeddings import EmbeddingManager
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: EmbeddingManager):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = config.TOP_K,
        min_score: float = config.MIN_SCORE,
    ) -> List[Dict[str, Any]]:
        """Chunks scoring below min_score are dropped. An empty list means
        'nothing relevant' - the pipeline then refuses instead of guessing."""
        query_embedding = self.embedder.embed_query(query)
        hits = self.vector_store.query(query_embedding, top_k)
        relevant = [h for h in hits if h["score"] >= min_score]
        logger.info(
            "Query %r: %d/%d chunks above %.2f (best %.3f)",
            query, len(relevant), len(hits), min_score, hits[0]["score"] if hits else 0.0,
        )
        return relevant
