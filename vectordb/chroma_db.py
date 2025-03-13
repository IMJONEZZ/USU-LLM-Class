"""
ChromaDB implementation of the vector database.
"""

import os
import uuid
import chromadb
from typing import Dict, List, Any, Optional

from zenml.logger import get_logger
from vectordb.base import VectorDatabase
from vectordb.embeddings import EmbeddingGenerator, DEFAULT_EMBEDDING_MODEL

logger = get_logger(__name__)

class ChromaVectorDB(VectorDatabase):
    """ChromaDB implementation of the vector database."""
    
    def __init__(
        self, 
        collection_name: str, 
        persist_directory: Optional[str] = None,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dimension: int = 0
    ):
        """Initialize the ChromaDB vector database.
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist the database (None for in-memory)
            embedding_model_name: Name of the model to use for embeddings
            embedding_dimension: Dimension of the embeddings (0 to auto-detect)
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Setup the embedding generator
        self.embedding_generator = EmbeddingGenerator(embedding_model_name)
        self.embedding_dimension = (
            embedding_dimension if embedding_dimension > 0 
            else self.embedding_generator.embedding_dimension
        )
        
        # Initialize ChromaDB client
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory)
            logger.info(f"Initialized persistent ChromaDB at {persist_directory}")
        else:
            self.client = chromadb.Client()
            logger.info("Initialized in-memory ChromaDB")
        
        # Create or get the collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except Exception as e:
            # Collection doesn't exist or other error, create a new one
            logger.info(f"Collection error: {str(e)}")
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_texts(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts to the vector database.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of ids for the texts
            
        Returns:
            List of ids for the added texts
        """
        if not texts:
            return []
            
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Generate ids if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            
        # Create default metadata if not provided
        if metadatas is None:
            # ChromaDB requires metadata dictionaries to be non-empty
            metadatas = [{"source": "default"} for _ in range(len(texts))]
            
        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List]:
        """Search the vector database for similar texts.
        
        Args:
            query: Query text
            n_results: Number of results to return
            where: Optional filtering criteria
            
        Returns:
            Dictionary with results including documents, metadatas, and distances
        """
        # Generate embedding for the query
        query_embedding = self.embedding_generator.generate_embeddings(query)
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=where
        )
        
        # Process results to ensure consistent format
        processed_results = {}
        
        # Handle documents
        if "documents" in results and results["documents"]:
            # Check if documents is a list of lists and flatten if needed
            if isinstance(results["documents"][0], list):
                processed_results["documents"] = results["documents"][0]
            else:
                processed_results["documents"] = results["documents"]
        else:
            processed_results["documents"] = []
            
        # Handle metadatas
        if "metadatas" in results and results["metadatas"]:
            # Check if metadatas is a list of lists and flatten if needed
            if isinstance(results["metadatas"][0], list):
                processed_results["metadatas"] = results["metadatas"][0]
            else:
                processed_results["metadatas"] = results["metadatas"]
        else:
            processed_results["metadatas"] = []
            
        # Handle distances
        if "distances" in results and results["distances"]:
            # Check if distances is a list of lists and flatten if needed
            if isinstance(results["distances"][0], list):
                processed_results["distances"] = results["distances"][0]
            else:
                processed_results["distances"] = results["distances"]
        else:
            processed_results["distances"] = []
            
        # Handle ids
        if "ids" in results and results["ids"]:
            # Check if ids is a list of lists and flatten if needed
            if isinstance(results["ids"][0], list):
                processed_results["ids"] = results["ids"][0]
            else:
                processed_results["ids"] = results["ids"]
        else:
            processed_results["ids"] = []
        
        return processed_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "count": count,
            "embedding_dimension": self.embedding_dimension,
            "embedding_model": self.embedding_generator.model_name,
        }
    
    def delete_collection(self) -> None:
        """Delete the collection."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")