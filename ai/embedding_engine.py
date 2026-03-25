"""BGE-M3 embedding engine via sentence-transformers.

Singleton model loader that generates 1024-dimensional dense
embeddings for all Milvus vector operations.  BGE-M3 is loaded
once on first use and cached for the process lifetime.

The embedding dimension (1024) MUST match the ``EMBEDDING_DIM``
constant in ``data.vector_store``.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """Singleton BGE-M3 embedding generator.

    Usage::

        engine = EmbeddingEngine()
        vec = engine.embed_text("hello world")
        assert len(vec) == 1024

        batch = engine.embed_batch(["hello", "world"])
        assert len(batch) == 2
    """

    _instance: ClassVar[EmbeddingEngine | None] = None
    _model: ClassVar[SentenceTransformer | None] = None

    def __new__(cls) -> EmbeddingEngine:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if EmbeddingEngine._model is None:
            logger.info(
                "Loading embedding model: %s",
                settings.EMBEDDING_MODEL,
            )
            EmbeddingEngine._model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
            )
            actual_dim = (
                EmbeddingEngine._model
                .get_sentence_embedding_dimension()
            )
            if actual_dim != settings.EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Embedding dimension mismatch: model "
                    f"produces {actual_dim}-dim vectors but "
                    f"config expects "
                    f"{settings.EMBEDDING_DIMENSION}. "
                    f"Update EMBEDDING_DIMENSION in config."
                )
            logger.info(
                "Embedding model loaded: %d-dim vectors",
                actual_dim,
            )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return settings.EMBEDDING_DIMENSION

    def embed_text(self, text: str) -> list[float]:
        """Generate an embedding for a single text string.

        Parameters
        ----------
        text : str
            Input text to embed.

        Returns
        -------
        list[float]
            1024-dimensional float vector.
        """
        vec = EmbeddingEngine._model.encode(
            text, normalize_embeddings=True,
        )
        return vec.tolist()

    def embed_batch(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Parameters
        ----------
        texts : list[str]
            Input texts.

        Returns
        -------
        list[list[float]]
            List of 1024-dimensional float vectors.
        """
        vecs = EmbeddingEngine._model.encode(
            texts, normalize_embeddings=True,
        )
        return vecs.tolist()
