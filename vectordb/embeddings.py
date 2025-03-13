"""
Utility functions for generating text embeddings.
"""

from typing import List, Union
import numpy as np

from sentence_transformers import SentenceTransformer

# Default model for embeddings
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingGenerator:
    """Class for generating embeddings from text."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """Initialize the embedding generator.

        Args:
            model_name: Name of the model to use for embeddings
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for the given texts.

        Args:
            texts: Text or list of texts to generate embeddings for

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embeddings.

        Returns:
            Dimension of the embeddings
        """
        return self.model.get_sentence_embedding_dimension()

    def __str__(self) -> str:
        return f"EmbeddingGenerator(model_name={self.model_name})"
