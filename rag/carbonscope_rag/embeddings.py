"""Step 3: turn text into vectors."""
import logging
from typing import List

import numpy as np

from . import config

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Wraps a SentenceTransformer model.

    - The model loads lazily on first use, so importing this module is cheap.
    - Vectors are L2-normalised, so dot product == cosine similarity and the
      scores we threshold on later are easy to interpret (1.0 = identical).
    """

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import, only when needed

            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_documents(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Embed many texts -> array of shape (len(texts), dim)."""
        return self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, show_progress_bar=show_progress
        )

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query -> array of shape (dim,)."""
        return self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
