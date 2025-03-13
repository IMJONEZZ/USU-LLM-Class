"""
Enhanced model evaluation with RAG support for Llama 3.2 fine-tuning
"""

import torch
from typing import Dict, Any, Optional
from typing_extensions import Annotated
from tqdm.auto import tqdm
from zenml import step
from zenml.logger import get_logger
from zenml import log_artifact_metadata

from config import TEST_QUESTIONS, MAX_NEW_TOKENS, HF_TOKEN, SYSTEM_PROMPT
from utils import (
    extract_structured_answer,
    strict_format_reward_func,
    soft_format_reward_func,
    use_unsloth,
)
from vectordb import ChromaVectorDB
from vectordb_step import retrieve_similar_context

logger = get_logger(__name__)


@step
def test_model_with_rag(
    model_info: Dict[str, Any],
    vectordb_info: Optional[Dict[str, Any]] = None,
    test_questions: list = TEST_QUESTIONS,
    max_new_tokens: int = MAX_NEW_TOKENS,
    hf_token: str = HF_TOKEN,
    use_rag: bool = True,
    n_results: int = 3,
    compare_with_without_rag: bool = True,
) -> Annotated[Dict[str, Any], "rag_evaluation_results"]:
    """Test the fine-tuned model on the assignment questions with optional RAG support.

    Args:
        model_info: Information about the trained model
        vectordb_info: Information about the vector database
        test_questions: List of test questions
        max_new_tokens: Maximum number of new tokens to generate
        hf_token: HuggingFace token
        use_rag: Whether to use RAG
        n_results: Number of results to retrieve for RAG
        compare_with_without_rag: Whether to compare results with and without RAG

    Returns:
        Dictionary with evaluation results
    """
    # First check if we can use Unsloth
    use_unsloth_backend = use_unsloth() and model_info.get("backend_used") == "unsloth"

    # Load model and tokenizer based on backend
    if use_unsloth_backend:
        try:
            # Only import unsloth if we're really going to use it
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_info["model_path"],
                max_seq_length=512,
                load_in_4bit=True,
                token=hf_token,
                gpu_memory_utilization=0.8,
            )
            # Prepare for inference
            model = FastLanguageModel.for_inference(model)
        except ImportError:
            logger.warning(
                "Failed to import unsloth, falling back to standard Transformers"
            )
            use_unsloth_backend = False

    # Standard transformers approach if unsloth failed or wasn't selected
    if not use_unsloth_backend:
        # Standard transformers approach for CPU
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_info["model_path"],
            token=hf_token,
        )

        # Load model with appropriate settings for CPU/GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_info["model_path"],
            token=hf_token,
            device_map="auto",  # Works for both CPU and GPU
            torch_dtype=torch.float32
            if not torch.cuda.is_available()
            else torch.float16,
            # Skip quantization when on CPU to avoid bitsandbytes issues
            load_in_8bit=False,
            quantization_config=None,
        )

        # Set padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    # Generation parameters
    generation_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
        "do_sample": True,
    }

    # Prepare results storage
    results = {
        "questions": test_questions,
        "answers": [],
        "formatted_qa_pairs": [],
        "rag_used": use_rag and vectordb_info is not None,
    }

    # Initialize vector database if RAG is enabled
    db = None
    if use_rag and vectordb_info is not None:
        db = ChromaVectorDB(
            collection_name=vectordb_info["collection_name"],
            persist_directory=vectordb_info["persist_directory"],
        )
        logger.info(
            f"Using RAG with vector database: {vectordb_info['collection_name']}"
        )

    # Test metrics
    format_compliance = []

    # Answer each question
    logger.info("Answering test questions...")

    for question in tqdm(test_questions):
        # For comparison, we'll generate answers both with and without RAG if requested
        answer_versions = ["with_rag"] if use_rag and db is not None else ["standard"]
        if compare_with_without_rag and use_rag and db is not None:
            answer_versions = ["standard", "with_rag"]

        answers_for_question = {}

        for version in answer_versions:
            is_rag = version == "with_rag"

            # Prepare prompt
            if is_rag:
                # Retrieve relevant context
                context_results = retrieve_similar_context(
                    question=question, vectordb_info=vectordb_info, n_results=n_results
                )
                context = context_results["retrieved_context"]

                # Format as chat with context
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Here is some relevant information:\n\n{context}\n\nWith this information, please answer the following question: {question}",
                    },
                ]
            else:
                # Standard prompt without RAG
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ]

            # Format prompt
            prompt = ""
            for msg in messages:
                prompt += f"{msg['role'].upper()}: {msg['content']}\n\n"

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # Generate answer
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generation_params,
                )

            # Decode the generated output
            generated_text = tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
            )

            # Extract reasoning and answer
            reasoning, answer = extract_structured_answer(generated_text)

            # Check format compliance
            strict_format = strict_format_reward_func(generated_text)
            soft_format = soft_format_reward_func(generated_text)

            # If no structured answer found, use the whole text
            if not answer:
                answer = generated_text

            # Store the answer version
            answers_for_question[version] = {
                "full_response": generated_text,
                "reasoning": reasoning,
                "final_answer": answer,
                "format_compliance_strict": strict_format,
                "format_compliance_soft": soft_format,
            }

            # For standard version, update metrics
            if version == "standard":
                format_compliance.append({"strict": strict_format, "soft": soft_format})

        # Add results to output
        result_entry = {
            "question": question,
            "standard": answers_for_question.get("standard", None),
            "with_rag": answers_for_question.get("with_rag", None),
            "rag_used": "with_rag" in answers_for_question,
        }

        results["answers"].append(result_entry)

        # Use the RAG answer if available, otherwise standard
        final_answer = (
            answers_for_question.get("with_rag", {})
            or answers_for_question.get("standard", {})
        ).get("final_answer", "")
        results["formatted_qa_pairs"].append(f"Q: {question}\nA: {final_answer}")

    # Calculate format compliance metrics for standard answers
    if format_compliance:
        strict_compliance_rate = sum(
            item["strict"] for item in format_compliance
        ) / len(format_compliance)
        soft_compliance_rate = sum(item["soft"] for item in format_compliance) / len(
            format_compliance
        )
    else:
        strict_compliance_rate = 0
        soft_compliance_rate = 0

    # Log metrics to ZenML
    evaluation_metrics = {
        "format_compliance_strict": strict_compliance_rate,
        "format_compliance_soft": soft_compliance_rate,
        "num_questions": len(test_questions),
        "backend_used": "unsloth" if use_unsloth_backend else "transformers",
        "rag_used": use_rag and vectordb_info is not None,
    }
    log_artifact_metadata(evaluation_metrics)

    # Save answers.txt file
    file_prefix = "rag_" if use_rag and vectordb_info is not None else ""
    with open(f"{file_prefix}answers.txt", "w") as f:
        for qa in results["formatted_qa_pairs"]:
            f.write(f"{qa}\n\n")

    logger.info(f"Answers saved to {file_prefix}answers.txt")
    logger.info(
        f"Format compliance: Strict={strict_compliance_rate:.2f}, Soft={soft_compliance_rate:.2f}"
    )

    if compare_with_without_rag and "with_rag" in answer_versions:
        logger.info("Generated answers with and without RAG for comparison")

    return results


def show_model_answers_with_rag(results):
    """Display the model's answers for all test questions, comparing with and without RAG."""
    for i, answer_data in enumerate(results["answers"]):
        question = answer_data["question"]
        print(f"\nQuestion {i + 1}: {question}")

        if answer_data.get("standard"):
            print("\nAnswer without RAG:")
            print(f"  {answer_data['standard']['final_answer']}")

        if answer_data.get("with_rag"):
            print("\nAnswer with RAG:")
            print(f"  {answer_data['with_rag']['final_answer']}")

        print("-" * 80)
