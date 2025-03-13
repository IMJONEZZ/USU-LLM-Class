"""
Tests for the vector database implementation.
"""

import uuid
import shutil
import tempfile
import pytest
import numpy as np

from vectordb import ChromaVectorDB, EmbeddingGenerator

class TestEmbeddingGenerator:
    """Tests for the EmbeddingGenerator class."""
    
    def test_initialization(self):
        """Test initialization of the EmbeddingGenerator."""
        generator = EmbeddingGenerator()
        assert generator.model_name == "all-MiniLM-L6-v2"
        assert generator.embedding_dimension > 0
        
    def test_generate_embeddings_single(self):
        """Test generating embeddings for a single text."""
        generator = EmbeddingGenerator()
        text = "This is a test document."
        
        embedding = generator.generate_embeddings(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, generator.embedding_dimension)
        
    def test_generate_embeddings_batch(self):
        """Test generating embeddings for multiple texts."""
        generator = EmbeddingGenerator()
        texts = ["This is document 1.", "This is document 2."]
        
        embeddings = generator.generate_embeddings(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, generator.embedding_dimension)


class TestChromaVectorDB:
    """Tests for the ChromaVectorDB class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for the tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Clean up after the tests
        shutil.rmtree(temp_dir)
    
    def test_initialization_in_memory(self):
        """Test initialization of in-memory ChromaVectorDB."""
        # Use a unique collection name to avoid conflicts
        collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        db = ChromaVectorDB(collection_name=collection_name)
        
        # Check basic properties
        assert db.collection_name == collection_name
        assert db.persist_directory is None
        assert db.embedding_dimension > 0
        
        # Check collection stats
        stats = db.get_collection_stats()
        assert stats["collection_name"] == collection_name
        assert stats["count"] == 0
        
        # Clean up
        db.delete_collection()
    
    def test_initialization_persistent(self, temp_dir):
        """Test initialization of persistent ChromaVectorDB."""
        # Use a unique collection name
        collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        db = ChromaVectorDB(
            collection_name=collection_name,
            persist_directory=temp_dir
        )
        
        # Check basic properties
        assert db.collection_name == collection_name
        assert db.persist_directory == temp_dir
        assert db.embedding_dimension > 0
        
        # Check collection stats
        stats = db.get_collection_stats()
        assert stats["collection_name"] == collection_name
        assert stats["count"] == 0
        
        # Clean up
        db.delete_collection()
    
    def test_add_texts(self):
        """Test adding texts to the database."""
        collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        db = ChromaVectorDB(collection_name=collection_name)
        
        texts = ["Document 1", "Document 2", "Document 3"]
        ids = db.add_texts(texts)
        
        # Check the number of added texts
        assert len(ids) == 3
        
        # Check collection stats
        stats = db.get_collection_stats()
        assert stats["count"] == 3
        
        # Clean up
        db.delete_collection()
    
    def test_add_texts_with_metadata(self):
        """Test adding texts with metadata to the database."""
        collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        db = ChromaVectorDB(collection_name=collection_name)
        
        texts = ["Document 1", "Document 2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        ids = db.add_texts(texts, metadatas=metadatas)
        
        # Check the number of added texts
        assert len(ids) == 2
        
        # Verify metadata was stored correctly
        results = db.search("Document", n_results=2)
        assert len(results["metadatas"]) == 2
        
        # Clean up
        db.delete_collection()
    
    def test_search(self):
        """Test searching the database."""
        collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        db = ChromaVectorDB(collection_name=collection_name)
        
        # Add some documents
        texts = [
            "Artificial intelligence is changing the world",
            "Vector databases enable semantic search",
            "Machine learning models require large datasets"
        ]
        # Add explicit metadatas to match ChromaDB's requirements
        metadatas = [
            {"source": "ai_doc"},
            {"source": "vector_doc"},
            {"source": "ml_doc"}
        ]
        db.add_texts(texts, metadatas=metadatas)
        
        # Search for relevant documents
        results = db.search("AI and machine learning", n_results=2)
        
        # Check the results
        assert len(results["documents"]) == 2
        assert len(results["distances"]) == 2
        
        # The first result should be the most relevant
        assert "Artificial" in results["documents"][0] or "Machine" in results["documents"][0]
        
        # Clean up
        db.delete_collection()
    
    def test_search_with_filter(self):
        """Test searching with filters."""
        collection_name = f"test_collection_{uuid.uuid4().hex[:8]}"
        db = ChromaVectorDB(collection_name=collection_name)
        
        # Add documents with metadata
        texts = [
            "Artificial intelligence research",
            "Vector database implementation",
            "Machine learning algorithms"
        ]
        metadatas = [
            {"category": "AI"},
            {"category": "Database"},
            {"category": "AI"}
        ]
        db.add_texts(texts, metadatas=metadatas)
        
        # Search with filter
        results = db.search(
            "algorithms and research",
            n_results=2,
            where={"category": "AI"}
        )
        
        # Check results
        assert len(results["documents"]) == 2
        
        # All results should be from AI category
        for metadata in results["metadatas"]:
            assert metadata["category"] == "AI"
        
        # Clean up
        db.delete_collection()