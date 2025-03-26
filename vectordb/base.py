"""
Base classes for vector database implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class VectorDatabase(ABC):
    """Abstract base class for vector database implementations."""

    @abstractmethod
    def __init__(self, collection_name: str, embedding_dimension: int = 768):
        """Initialize the vector database.

        Args:
            collection_name: Name of the collection to use
            embedding_dimension: Dimension of the embeddings
        """
        pass

    @abstractmethod
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add texts to the vector database.

        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of ids for the texts

        Returns:
            List of ids for the added texts
        """
        pass

    @abstractmethod
    def search(
        self, query: str, n_results: int = 5, where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List]:
        """Search the vector database for similar texts.

        Args:
            query: Query text
            n_results: Number of results to return
            where: Optional filtering criteria

        Returns:
            Dictionary with results including documents, metadatas, and distances
        """
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection statistics
        """
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the collection."""
        pass
