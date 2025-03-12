"""
Configuration settings for Llama 3.2 fine-tuning
"""

import os

# HuggingFace settings
HUGGINGFACE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # Set your token as an environment variable for security

# ZenML settings
ZENML_SERVER_URL = os.environ.get("ZENML_SERVER_URL", "")  # Set your ZenML server URL as an environment variable

# Model training parameters
MAX_SEQ_LENGTH = 512
LORA_RANK = 16
NUM_TRAIN_EPOCHS = 3
LEARNING_RATE = 2e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
OUTPUT_DIR = "llama_finetuned"

# Testing parameters
MAX_NEW_TOKENS = 256

# System prompt for the model
SYSTEM_PROMPT = """You are a helpful, harmless, and precise assistant. When responding, first think through the problem step-by-step, then provide your final answer.

Respond in the following format:
<reasoning>
[Your step-by-step thinking about the problem]
</reasoning>
<answer>
[Your final, concise answer]
</answer>
"""

# Test questions for evaluation
TEST_QUESTIONS = [
    "If it takes 1 hour for 60 people to play an Opera, how many hours will it take 600 people to play the same opera?",
    "Is a pound of feathers or a British pound heavier?",
    "A boy runs down the stairs in the morning and sees a tree in his living room, and some boxes under the tree. What's going on?",
    "What happens if you crack your knuckles a lot?",
    "If there is a shark in the pool of my basement, is it safe to go upstairs?",
    "How much wood could a wood chuck chuck if there were only 5 pounds of wood in the world?",
    "Who is the current President of the United States?",
    "Was Talos alive?",
    "How many Ls are in the word parallel?",
    "What is the riddle of the sphinx, and what are all possible answers satisfying all conditions?"
]