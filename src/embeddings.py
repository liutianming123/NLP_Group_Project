"""Embedding generation for semantic search."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import app_config

logger = logging.getLogger(__name__)


class VectorizationService:
    """Service for generating text embeddings."""

    def __init__(self) -> None:
        """Initialize the embedding model."""
        self.model_name = app_config.embed_model
        self.device = app_config.embed_device
        self.model: SentenceTransformer | None = None
        self.embedding_dim = 768  # Multilingual mpnet uses 768 dimensions

    def initialize_transformer(self) -> None:
        """Load the sentence-transformer model."""
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Transformer model loaded. Dimension: {self.embedding_dim}")

    def generate_embedding(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to encode

        Returns:
            List of floats representing the embedding vector
        """
        if self.model is None:
            self.initialize_transformer()

        assert self.model is not None
        embedding_vector = self.model.encode(text, convert_to_numpy=True)
        return embedding_vector.tolist()

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to encode

        Returns:
            List of embedding vectors
        """
        if self.model is None:
            self.initialize_transformer()

        assert self.model is not None
        embedding_vectors = self.model.encode(
            texts, batch_size=app_config.batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return embedding_vectors.tolist()

    def calculate_cosine_similarity(self, emb1: list[float], emb2: list[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Similarity score between 0.0 and 1.0
        """
        vec1 = np.array(emb1)
        vec2 = np.array(emb2)

        dot_prod = np.dot(vec1, vec2)
        norm_v1 = np.linalg.norm(vec1)
        norm_v2 = np.linalg.norm(vec2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        similarity_score = dot_prod / (norm_v1 * norm_v2)
        return float(similarity_score)


# Global embedding service instance
vectorizer = VectorizationService()