"""
Load test questions from config.py into the vector database.
"""

from typing import List, Dict
from zenml.logger import get_logger

from config import TEST_QUESTIONS
from vectordb import ChromaVectorDB

logger = get_logger(__name__)

def load_test_questions(
    collection_name: str = "test_questions",
    persist_directory: str = "./chroma_data"
) -> ChromaVectorDB:
    """Load test questions into a vector database.
    
    Args:
        collection_name: Name of the collection to use
        persist_directory: Directory to persist the database
        
    Returns:
        ChromaVectorDB instance with the loaded questions
    """
    # Initialize the vector database
    db = ChromaVectorDB(
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    # Prepare metadata for each question (ChromaDB requires non-empty metadata)
    metadatas = [{"question_id": i, "type": "test_question"} for i in range(len(TEST_QUESTIONS))]
    
    # Add questions to the database
    logger.info(f"Loading {len(TEST_QUESTIONS)} test questions into vector database")
    ids = db.add_texts(TEST_QUESTIONS, metadatas=metadatas)
    
    logger.info(f"Successfully loaded {len(ids)} questions")
    stats = db.get_collection_stats()
    logger.info(f"Collection stats: {stats}")
    
    return db

def query_similar_questions(
    query: str,
    db: ChromaVectorDB,
    n_results: int = 3
) -> List[Dict]:
    """Query the database for similar questions.
    
    Args:
        query: Query text
        db: ChromaVectorDB instance
        n_results: Number of results to return
        
    Returns:
        List of result dictionaries
    """
    results = db.search(query, n_results=n_results)
    
    formatted_results = []
    for i in range(len(results["documents"])):
        formatted_results.append({
            "question": results["documents"][i],
            "metadata": results["metadatas"][i],
            "distance": results["distances"][i]
        })
    
    return formatted_results

def main():
    """Main function to demonstrate loading and querying test questions."""
    # Load questions into the database
    db = load_test_questions()
    
    # Example queries
    queries = [
        "What happens if I crack my fingers?",
        "Is it safe to go upstairs if there's a shark in the basement?",
        "How many letters are in a word?",
        "Who is the current president?",
        "What is the sphinx riddle?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        results = query_similar_questions(query, db)
        
        print(f"Found {len(results)} similar questions:")
        for i, result in enumerate(results):
            print(f"  Result {i+1}:")
            print(f"    Question: {result['question']}")
            print(f"    Distance: {result['distance']:.4f}")
            print(f"    Metadata: {result['metadata']}")

if __name__ == "__main__":
    main()