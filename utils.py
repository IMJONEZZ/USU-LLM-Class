"""
Utility functions for Llama 3.2 fine-tuning
"""

import re
import os
import torch
from zenml.client import Client
from huggingface_hub import login

# Global flag to force CPU mode - set by run.py when --force-cpu is passed
FORCE_CPU_MODE = False

def setup_environment(zenml_server_url=None, hf_token=None):
    """
    Set up the environment for training, including logging into ZenML and HuggingFace
    """
    # Use environment variables if parameters are not provided
    zenml_server_url = zenml_server_url or os.environ.get("ZENML_SERVER_URL")
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    
    # Validate inputs
    if not zenml_server_url:
        raise ValueError("ZenML server URL is required. Set ZENML_SERVER_URL environment variable or pass as parameter.")
    if not hf_token:
        raise ValueError("HuggingFace token is required. Set HF_TOKEN environment variable or pass as parameter.")
    
    # Disable wandb
    os.environ["WANDB_DISABLED"] = "true"
    
    # Initialize ZenML
    os.system(f"zenml login {zenml_server_url}")
    os.system("zenml init")
    os.system("zenml stack set default")
    
    # Initialize ZenML client
    client = Client()
    print(f"Connected to ZenML server at {client.zen_store.url}")
    
    # Log in to HuggingFace
    login(token=hf_token)
    print("Successfully logged in to Hugging Face")
    
    return client

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

def check_gpu():
    """Check if a GPU is available and print details."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"Found {device_count} GPU(s): {device_name}")
        return "cuda"
    else:
        print("No GPU found, using CPU")
        return "cpu"
        
def use_unsloth():
    """Determine whether to use Unsloth based on hardware.
    This function NEVER imports unsloth directly to avoid GPU checks.
    
    Returns:
        bool: True if Unsloth should be used (CUDA available), False otherwise
    """
    # Check the global flag first
    global FORCE_CPU_MODE
    if FORCE_CPU_MODE:
        print("Force CPU mode is enabled, not using Unsloth")
        return False
    
    # Check environment variable for GPU visibility
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "-1":
        print("CUDA_VISIBLE_DEVICES=-1, not using Unsloth")
        return False
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("GPU not available, not using Unsloth")
        return False
    
    # Only try to check if unsloth is available without importing it
    try:
        # Use importlib.util to check if module exists without importing
        import importlib.util
        if importlib.util.find_spec("unsloth") is not None:
            print("Unsloth is available and GPU detected, will use Unsloth")
            return True
        else:
            print("Unsloth module not found, using standard Transformers")
            return False
    except Exception:
        print("Error checking for Unsloth, using standard Transformers")
        return False