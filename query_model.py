"""
Script to query the fine-tuned Llama model with RAG support
"""

import os
import argparse
import torch
from zenml.logger import get_logger
from utils import extract_structured_answer, check_gpu
from config import SYSTEM_PROMPT, MAX_NEW_TOKENS
from vectordb_step import retrieve_similar_context

logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Query the fine-tuned Llama model with RAG support")
    
    # Model configuration
    parser.add_argument("--model-path", type=str, default="llama_finetuned", 
                        help="Path to the fine-tuned model")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    
    # Vector database configuration
    parser.add_argument("--vectordb-collection", type=str, default="llama_knowledge", 
                        help="Vector database collection name")
    parser.add_argument("--vectordb-dir", type=str, default="./chroma_data", 
                        help="Vector database persistence directory")
    
    # Query configuration
    parser.add_argument("--question", type=str, required=True,
                        help="Question to ask the model")
    parser.add_argument("--use-rag", action="store_true", default=True,
                        help="Use RAG during querying")
    parser.add_argument("--no-rag", action="store_false", dest="use_rag",
                        help="Don't use RAG during querying")
    parser.add_argument("--rag-results", type=int, default=3,
                        help="Number of results to retrieve for RAG")
    parser.add_argument("--compare", action="store_true",
                        help="Compare results with and without RAG")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for text generation")
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Maximum number of tokens to generate")
    
    # Backend selection (force CPU if needed)
    parser.add_argument("--force-cpu", action="store_true", 
                        help="Force CPU usage even if GPU is available")
    
    return parser.parse_args()

def load_model(model_path, hf_token=None):
    """Load the model and tokenizer."""
    # Use standard transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Get HF token from environment if not provided
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        token=hf_token,
    )
    
    # Set padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        token=hf_token,
        device_map="auto",
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        load_in_8bit=False,
        quantization_config=None
    )
    
    return model, tokenizer

def query_model(
    question, 
    model, 
    tokenizer, 
    vectordb_info=None, 
    use_rag=True,
    rag_results=3,
    temperature=0.7,
    max_tokens=MAX_NEW_TOKENS
):
    """Query the model with or without RAG."""
    # Generation parameters
    generation_params = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 50,
        "do_sample": True,
    }
    
    # Prepare prompt
    if use_rag and vectordb_info is not None:
        # Retrieve relevant context
        context_results = retrieve_similar_context(
            question=question,
            vectordb_info=vectordb_info,
            n_results=rag_results
        )
        context = context_results["retrieved_context"]
        
        # Show retrieved context
        print("\nRetrieved context:")
        print(context)
        
        # Format as chat with context
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here is some relevant information:\n\n{context}\n\nWith this information, please answer the following question: {question}"}
        ]
    else:
        # Standard prompt without RAG
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
    print("\nGenerating answer...")
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
    
    # If no structured answer found, use the whole text
    if not answer:
        answer = generated_text
    
    return {
        "full_response": generated_text,
        "reasoning": reasoning,
        "answer": answer,
        "rag_used": use_rag and vectordb_info is not None
    }

def main():
    """Main function to query the model."""
    args = parse_args()
    
    # Get HF token from environment if not provided
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    
    # Set environment variables for later use
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
    
    # Handle forced CPU usage if requested
    if args.force_cpu:
        logger.info("Forcing CPU usage as requested")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Hide GPU from PyTorch
    
    # Check for GPU
    check_gpu()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path, hf_token)
    
    # Prepare vector database info if RAG is enabled
    vectordb_info = None
    if args.use_rag:
        vectordb_info = {
            "collection_name": args.vectordb_collection,
            "persist_directory": args.vectordb_dir
        }
        print(f"Using RAG with vector database: {args.vectordb_collection}")
    
    # Query the model
    print(f"Question: {args.question}")
    
    # Compare with and without RAG if requested
    if args.compare:
        # Without RAG
        print("\n--- Answer without RAG ---")
        result_without_rag = query_model(
            question=args.question,
            model=model,
            tokenizer=tokenizer,
            use_rag=False,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        print("\nReasoning:")
        print(result_without_rag["reasoning"])
        print("\nAnswer:")
        print(result_without_rag["answer"])
        
        # With RAG
        if vectordb_info is not None:
            print("\n--- Answer with RAG ---")
            result_with_rag = query_model(
                question=args.question,
                model=model,
                tokenizer=tokenizer,
                vectordb_info=vectordb_info,
                use_rag=True,
                rag_results=args.rag_results,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )
            
            print("\nReasoning:")
            print(result_with_rag["reasoning"])
            print("\nAnswer:")
            print(result_with_rag["answer"])
    else:
        # Regular query
        result = query_model(
            question=args.question,
            model=model,
            tokenizer=tokenizer,
            vectordb_info=vectordb_info if args.use_rag else None,
            use_rag=args.use_rag,
            rag_results=args.rag_results,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        print("\nReasoning:")
        print(result["reasoning"])
        print("\nAnswer:")
        print(result["answer"])

if __name__ == "__main__":
    main()