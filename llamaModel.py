from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# Hugging Face authentication token

with open('keys.json') as f:
    keys = json.load(f)
token = keys['huggingfaceToken']

# Load Llama 2 or Mistral model
model_name = "meta-llama/Llama-3.2-1B" 

# Set device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map=device, token=token
)

QUESTIONS = [
    "If it takes 1 hour for 60 people to play an Opera, how many hours will it take 600 people to play the same opera?",
    "Is a pound of feathers or a British pound heavier?",
    "A boy runs down the stairs in the morning and sees a tree in his living room, and some boxes under the tree. What's going on?",
    "What happens if you crack your knuckles a lot?",
    "If there is a shark in the pool of my basement, is it safe to go upstairs?",
    "How much wood could a wood chuck chuck if there were only 5 pounds of wood in the world?",
    "Who is the current President of the United States?",
    "Was Talos alive?",
    "How many Ls are in the word 'parallel'?",
    "What is the riddle of the sphinx, and what are all possible answers satisfying all conditions?",
]

# Function to generate responses
def generate_response(question):
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
    output = model.generate(input_ids, max_length=100)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Save answers to file
output_file = "answersWithoutAgent.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for question in QUESTIONS:
        answer = generate_response(question)
        f.write(f"Q: {question}\nA: {answer}\n\n")
        print(f"Q: {question}\nA: {answer}\n")  # Print to console as well

print(f"\n✅ Answers saved in {output_file}")




