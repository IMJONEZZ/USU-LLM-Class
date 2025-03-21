import pytest
from unittest.mock import MagicMock
import numpy as np
import sys

sys.path.append('/home/TomKerby/School/USU-LLM-Class/')
from wiki_data_digestion import WikiDataIngestion
from sentence_transformers import SentenceTransformer

@pytest.fixture(scope="module")
def mock_pinecone():
    mock_index = MagicMock()
    mock_index.describe_index_stats.return_value = {"total_vector_count": 0}
    mock_index.upsert.return_value = {"upserted_count": 42}
    mock_index.query.return_value = {
        "matches": [
            {"id": "1", "score": 0.81, "metadata": {"text": "Example match 1"}},
            {"id": "2", "score": 0.67, "metadata": {"text": "Example match 2"}},
        ]
    }
    return mock_index

@pytest.fixture
def mock_wiki_dataset():
    return MagicMock(
        __iter__=lambda _: iter([
            {
                "id": "1",
                "text": "Gutenberg printing press history",
                "url": "http://example.com/gutenberg",
                "title": "Printing Press"
            },
            {
                "id": "2",
                "text": "Space exploration facts",
                "url": "http://example.com/space",
                "title": "Space Exploration"
            }
        ])
    )

@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.encode.return_value = np.random.rand(384).astype(np.float32)  # Return a 384-dim vector
    return embedder

def test_class_initialization_defaults(mock_pinecone):
    """Test initialization with default parameters"""
    ingestion = WikiDataIngestion(mock_pinecone)
    
    assert ingestion.batch_limit == 100
    assert isinstance(ingestion.embedder, SentenceTransformer)
    assert ingestion.text_splitter._chunk_size == 200

def test_text_splitting_and_metadata():
    """Test text splitting and metadata generation"""
    ingestion = WikiDataIngestion(MagicMock())
    page = {
        "id": "123",
        "text": "This is a test text. It should be split into multiple chunks. if I make it suffciently long",
        "url": "http://test.com",
        "title": "Test Page"
    }
    
    texts, metadatas = ingestion.split_texts_and_metadatas(page)
    
    assert len(texts) == len(metadatas)
    assert all(isinstance(t, str) for t in texts)
    assert all(m["wiki-id"] == "123" for m in metadatas)
    assert all(m["source"] == page["url"] for m in metadatas)

def test_batch_upload_logic(mock_pinecone, mock_wiki_dataset, mock_embedder):
    ingestion = WikiDataIngestion(
        mock_pinecone,
        wikidata=mock_wiki_dataset,
        embedder=mock_embedder,  # Inject mock embedder
        batch_limit=1
    )
    
    ingestion.batch_upload()

    assert mock_embedder.encode.call_count == 2  # 2 pages * 1 chunk each
    assert mock_pinecone.upsert.call_count == 2

def test_upsert_batch_content(mock_pinecone, mock_embedder):
    """Test vector format and metadata in upsert calls"""
    ingestion = WikiDataIngestion(
        mock_pinecone,
        embedder=mock_embedder,
        batch_limit=2
    )
    
    texts = ["text1", "text2"]
    metadatas = [{"meta": "1"}, {"meta": "2"}]
    
    ingestion.upload_batch(texts, metadatas)
    
    # Verify vector structure
    mock_pinecone.upsert.call_count == 3
    args, kwargs = mock_pinecone.upsert.call_args
    vectors = list(kwargs.get('vectors'))
    
    assert len(vectors) == 2
    assert all(len(v) == 3 for v in vectors)  # (id, embedding, metadata)
    assert vectors[0][2]["meta"] == "1"
    assert vectors[1][2]["meta"] == "2"


def test_query_execution(mock_pinecone, mock_embedder):
    """Test full query workflow with mocked components"""
    ingestion = WikiDataIngestion(mock_pinecone, embedder=mock_embedder)
    test_query = "Test query"
    
    # Execute query
    embeddings = ingestion.embedder.encode(test_query).tolist()
    results = ingestion.index.query(
        vector=embeddings,
        top_k=3,
        include_metadata=True
    )
    
    # Verify call arguments
    mock_pinecone.query.assert_called_once_with(
        vector=embeddings,
        top_k=3,
        include_metadata=True
    )
    assert len(results["matches"]) == 2
    assert all(m["score"] > 0 for m in results["matches"])