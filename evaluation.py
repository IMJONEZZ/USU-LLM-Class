"""
Model evaluation functionality for Llama 3.2 fine-tuning
"""

import torch
from typing import Dict, Any
from typing_extensions import Annotated
from tqdm.auto import tqdm
from zenml import step
from zenml.logger import get_logger
from zenml import log_artifact_metadata

from config import TEST_QUESTIONS, MAX_NEW_TOKENS, HF_TOKEN, SYSTEM_PROMPT
from utils import extract_structured_answer, strict_format_reward_func, soft_format_reward_func, use_unsloth

logger = get_logger(__name__)

@step
def test_model(
    model_info: Dict[str, Any],
    test_questions: list = TEST_QUESTIONS,
    max_new_tokens: int = MAX_NEW_TOKENS,
    hf_token: str = HF_TOKEN
) -> Annotated[Dict[str, Any], "evaluation_results"]:
    """Test the fine-tuned model on the assignment questions."""
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
            logger.warning("Failed to import unsloth, falling back to standard Transformers")
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
            torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16,
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
                attention_mask=inputs.attention_mask,
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
        "backend_used": "unsloth" if use_unsloth_backend else "transformers",
    }
    log_artifact_metadata(evaluation_metrics)

    # Save answers.txt file
    with open("answers.txt", "w") as f:
        for qa in results["formatted_qa_pairs"]:
            f.write(f"{qa}\n\n")

    logger.info(f"Answers saved to answers.txt")
    logger.info(f"Format compliance: Strict={strict_compliance_rate:.2f}, Soft={soft_compliance_rate:.2f}")

    return results

def show_model_answers(results):
    """Display the model's answers for all test questions."""
    for i, (question, answer) in enumerate(zip(results["questions"], results["answers"])):
        print(f"Question {i+1}: {question}")
        print(f"Answer: {answer['final_answer']}")
        print("-" * 80)