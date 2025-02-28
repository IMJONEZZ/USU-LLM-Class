from typing import Dict, List, Any
from typing_extensions import Annotated
import re
import os
import torch
from tqdm.auto import tqdm
from zenml import step, log_artifact_metadata
from zenml.logger import get_logger

logger = get_logger(__name__)

def extract_structured_answer(text):
    """Extract reasoning and answer sections from a structured response."""
    # Default values
    reasoning = ""
    answer = ""

    # Extract reasoning section
    reasoning_pattern = r"<reasoning>(.*?)</reasoning>"
    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Extract answer section
    answer_pattern = r"<answer>(.*?)</answer>"
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()

    # If no structured format found, use the entire text as the answer
    if not reasoning and not answer:
        answer = text.strip()

    return reasoning, answer

# Define reward functions for format compliance
def strict_format_reward_func(text):
    """Reward function that checks if the text has the expected structure."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    match = re.match(pattern, text, re.DOTALL)
    return 1.0 if match else 0.0

def soft_format_reward_func(text):
    """Reward function with more lenient formatting requirements."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    match = re.search(pattern, text, re.DOTALL)
    return 1.0 if match else 0.0

@step
def test_model(
    model_info: Dict[str, Any],
    test_questions: List[str],
    hf_token: str = None,
    max_new_tokens: int = 256
) -> Annotated[Dict[str, Any], "evaluation_results"]:
    """Test the fine-tuned model on the assignment questions.
    
    This step evaluates the fine-tuned LLaMA model on a set of test questions
    and generates answers in the structured format.
    
    Args:
        model_info: Dictionary with model information
        test_questions: List of test questions to evaluate
        hf_token: HuggingFace API token
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Dict with evaluation results and model answers
    """
    # Validate HF token
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token is None:
            raise ValueError(
                "HuggingFace token is required. Please provide it via the 'hf_token' argument "
                "or set the 'HF_TOKEN' environment variable."
            )
            
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        logger.error("Unsloth is not installed. Please run 'pip install unsloth'")
        raise

    # System prompt for our model
    SYSTEM_PROMPT = """You are a helpful, harmless, and precise assistant. When responding, first think through the problem step-by-step, then provide your final answer.

    Respond in the following format:
    <reasoning>
    [Your step-by-step thinking about the problem]
    </reasoning>
    <answer>
    [Your final, concise answer]
    </answer>
    """

    # Load the model from the saved path
    logger.info(f"Loading model from {model_info['model_path']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_info["model_path"],
        max_seq_length=512,
        load_in_4bit=True,
        token=hf_token,
        gpu_memory_utilization=0.8,
    )

    # Prepare the model for inference
    model = FastLanguageModel.for_inference(model)

    # Generation parameters
    generation_params = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 50,
    }

    # Prepare results storage
    results = {
        "questions": test_questions,
        "answers": [],
        "formatted_qa_pairs": []
    }

    # Test metrics
    format_compliance = []

    # Answer each question
    logger.info("Answering test questions...")

    for question in tqdm(test_questions):
        # Format as chat
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
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
                **generation_params
            )

        # Decode the generated output
        generated_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        # Extract reasoning and answer
        reasoning, answer = extract_structured_answer(generated_text)

        # Check format compliance
        strict_format = strict_format_reward_func(generated_text)
        soft_format = soft_format_reward_func(generated_text)
        format_compliance.append({"strict": strict_format, "soft": soft_format})

        # If no structured answer found, use the whole text
        if not answer:
            answer = generated_text

        # Add to results
        results["answers"].append({
            "question": question,
            "full_response": generated_text,
            "reasoning": reasoning,
            "final_answer": answer,
            "format_compliance_strict": strict_format,
            "format_compliance_soft": soft_format,
        })

        # Add to formatted Q&A pairs
        results["formatted_qa_pairs"].append(f"Q: {question}\nA: {answer}")

    # Calculate format compliance metrics
    strict_compliance_rate = sum(item["strict"] for item in format_compliance) / len(format_compliance)
    soft_compliance_rate = sum(item["soft"] for item in format_compliance) / len(format_compliance)

    # Log metrics to ZenML
    evaluation_metrics = {
        "format_compliance_strict": strict_compliance_rate,
        "format_compliance_soft": soft_compliance_rate,
        "num_questions": len(test_questions),
    }
    log_artifact_metadata(evaluation_metrics)

    # Save answers.txt file
    with open("answers.txt", "w") as f:
        for qa in results["formatted_qa_pairs"]:
            f.write(f"{qa}\n\n")

    logger.info(f"Answers saved to answers.txt")
    logger.info(f"Format compliance: Strict={strict_compliance_rate:.2f}, Soft={soft_compliance_rate:.2f}")

    return results