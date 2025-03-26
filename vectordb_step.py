"""
Vector database integration step for the Llama 3.2 fine-tuning pipeline.
"""

from typing import Dict, Any, List, Optional
from typing_extensions import Annotated
from datasets import Dataset
from zenml import step
from zenml.logger import get_logger
from zenml import log_metadata

from config import TEST_QUESTIONS
from vectordb import ChromaVectorDB

logger = get_logger(__name__)


@step
def create_vector_database(
    dataset: Optional[Dataset] = None,
    test_questions: List[str] = TEST_QUESTIONS,
    collection_name: str = "llama_knowledge",
    persist_directory: str = "./chroma_data",
    include_sample_data: bool = True,
    include_test_questions: bool = True,
) -> Annotated[Dict[str, Any], "vectordb_info"]:
    """Create and populate the vector database with training data and test questions.

    Args:
        dataset: The training dataset
        test_questions: The test questions
        collection_name: Name of the vector database collection
        persist_directory: Directory to persist the vector database
        include_sample_data: Whether to include sample data in the vector database
        include_test_questions: Whether to include test questions in the vector database

    Returns:
        Dictionary with vector database information
    """
    # Create the vector database
    logger.info(f"Creating vector database collection '{collection_name}'")
    db = ChromaVectorDB(
        collection_name=collection_name, persist_directory=persist_directory
    )

    # Track the number of documents added
    doc_count = 0

    # Add training data if available and requested
    if dataset is not None and include_sample_data:
        texts = []
        metadatas = []

        # Extract information from the dataset
        for i, example in enumerate(dataset):
            # Get the instruction/question
            user_message = [
                msg for msg in example["messages"] if msg["role"] == "user"
            ][0]
            question = user_message["content"]

            # Get the target/answer
            response = example["target"]

            # Add both question and response to the database
            texts.append(question)
            metadatas.append(
                {"source": "training_data", "example_id": i, "type": "question"}
            )

            texts.append(response)
            metadatas.append(
                {
                    "source": "training_data",
                    "example_id": i,
                    "type": "response",
                    "for_question": question,
                }
            )

        # Add to vector database
        logger.info(f"Adding {len(texts)} training data documents to vector database")
        db.add_texts(texts, metadatas=metadatas)
        doc_count += len(texts)

    # Add test questions if requested
    if include_test_questions:
        # Prepare metadata for test questions
        question_metadatas = [
            {"source": "test_questions", "question_id": i, "type": "question"}
            for i in range(len(test_questions))
        ]

        # Add to vector database
        logger.info(f"Adding {len(test_questions)} test questions to vector database")
        db.add_texts(test_questions, metadatas=question_metadatas)
        doc_count += len(test_questions)

    # Get database stats
    stats = db.get_collection_stats()
    logger.info(f"Vector database stats: {stats}")

    # Log metrics
    log_metadata(
        {
            "vectordb_collection": collection_name,
            "vectordb_doc_count": doc_count,
            "vectordb_embedding_dimension": stats["embedding_dimension"],
            "vectordb_embedding_model": stats["embedding_model"],
        }
    )

    # Return vector database information
    return {
        "collection_name": collection_name,
        "persist_directory": persist_directory,
        "doc_count": doc_count,
        "embedding_dimension": stats["embedding_dimension"],
        "embedding_model": stats["embedding_model"],
    }


@step
def retrieve_similar_context(
    question: str, vectordb_info: Dict[str, Any], n_results: int = 3
) -> Dict[str, Any]:
    """Retrieve similar context from the vector database for a question.

    Args:
        question: The question to find similar context for
        vectordb_info: Vector database information
        n_results: Number of results to retrieve

    Returns:
        Dictionary with retrieved context
    """
    # Initialize the vector database from the provided info
    db = ChromaVectorDB(
        collection_name=vectordb_info["collection_name"],
        persist_directory=vectordb_info["persist_directory"],
    )

    # Retrieve similar documents
    results = db.search(question, n_results=n_results)

    # Format the results differently
    retrieved_context = (
        "Here is some relevant information that might help with your answer:\n\n"
    )

    # Track what we've already included to avoid duplication
    included_docs = set()

    # Keep track of processed entries for retrieved_texts
    retrieved_texts = []

    # First pass: prefer full answers
    for i, (doc, metadata) in enumerate(
        zip(results["documents"], results["metadatas"])
    ):
        # Skip questions since we already have the question
        if metadata.get("type") == "question":
            continue

        # Prioritize full answers
        if metadata.get("type") == "full_answer":
            if doc not in included_docs:
                retrieved_context += f"{doc}\n\n"
                included_docs.add(doc)

        # Add to retrieved_texts regardless
        retrieved_texts.append(
            {"text": doc, "metadata": metadata, "distance": results["distances"][i]}
        )

    # Second pass: include other relevant info if we don't have enough context
    if len(included_docs) < 2:
        for i, (doc, metadata) in enumerate(
            zip(results["documents"], results["metadatas"])
        ):
            if doc not in included_docs and metadata.get("type") != "question":
                retrieved_context += f"{doc}\n\n"
                included_docs.add(doc)
                if len(included_docs) >= 2:
                    break

    return {
        "question": question,
        "retrieved_context": retrieved_context,
        "retrieved_texts": retrieved_texts,
        "raw_results": results,
    }
