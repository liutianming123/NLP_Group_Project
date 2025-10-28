"""Embedding generation for semantic search."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(self) -> None:
        """Initialize the embedding model."""
        self.model_name = settings.embed_model
        self.device = settings.embed_device
        self.model: SentenceTransformer | None = None
        self.embedding_dim = 768  # Multilingual mpnet uses 768 dimensions

    def load_model(self) -> None:
        """Load the sentence-transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def encode(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to encode

        Returns:
            List of floats representing the embedding vector
        """
        if self.model is None:
            self.load_model()

        assert self.model is not None
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to encode

        Returns:
            List of embedding vectors
        """
        if self.model is None:
            self.load_model()

        assert self.model is not None
        embeddings = self.model.encode(
            texts, batch_size=settings.batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return embeddings.tolist()

    def cosine_similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0.0 and 1.0
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        similarity = dot_product / (norm1 * norm2)
        return float(similarity)


# Global embedding service instance
embedding_service = EmbeddingService()
