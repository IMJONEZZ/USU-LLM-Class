from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

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

# Custom knowledge base
CUSTOM_KNOWLEDGE_BASE = {
    "opera_time": "If it takes 1 hour for 60 people to play an opera, adding more people does not change the time required. The performance length stays the same, so 600 people will also take 1 hour.",
    "feathers_vs_pound": "A British pound (currency) is a fixed unit of money, while a pound of feathers is a measure of weight. The pound of feathers is heavier.",
    "christmas_tree": "A tree in the living room with boxes underneath usually indicates Christmas morning, with presents placed under the tree.",
    "knuckle_cracking": "Cracking knuckles releases gas bubbles in the joints. It does not cause arthritis but may lead to reduced grip strength over time.",
    "shark_in_pool": "If there is a shark in the basement pool, it is irrelevant to going upstairs unless the shark has supernatural powers or can leave the water.",
    "woodchuck_riddle": "If only 5 pounds of wood exist in the world, a woodchuck could chuck at most 5 pounds of wood.",
    "us_president": "The current President of the United States is Joe Biden (as of 2024).",
    "talos_myth": "Talos is a figure from Greek mythology, a giant bronze automaton created to protect Crete.",
    "parallel_spelling": "The word 'parallel' contains 3 'L's.",
    "sphinx_riddle": "The riddle of the Sphinx: 'What walks on four legs in the morning, two in the afternoon, and three in the evening?' Answer: A human (crawling as a baby, walking as an adult, using a cane when old)."
}

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Create FAISS vector database
knowledge_texts = list(CUSTOM_KNOWLEDGE_BASE.values())
knowledge_keys = list(CUSTOM_KNOWLEDGE_BASE.keys())

# Encode knowledge base
embeddings = embedder.encode(knowledge_texts)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Mapping index to text
id_to_text = {i: knowledge_texts[i] for i in range(len(knowledge_texts))}

# Function to retrieve the most relevant stored knowledge
def retrieve_knowledge(question, top_k=2, threshold=0.7):
    question_embedding = embedder.encode([question])
    distances, indices = index.search(np.array(question_embedding), top_k)
    
    best_match_id = indices[0][0]
    best_match_distance = distances[0][0]

    if best_match_distance < threshold:  # Lower means more relevant
        return id_to_text[best_match_id]
    
    return None

# Function to generate response
def generate_response(question):
    retrieved_knowledge = retrieve_knowledge(question)

    if retrieved_knowledge:
        return f"{retrieved_knowledge} (Retrieved from knowledge base.)"

    # If no relevant knowledge is found, use LLM
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)
    output = model.generate(
        input_ids, 
        max_length=100, 
        temperature=0.7,  # Reduce randomness
        top_p=0.9,  # Control diversity
        repetition_penalty=1.2  # Reduce repetition
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    
    # Prevent excessive repetition
    response = response.split("\n")[0]
    
    return response

# List of questions
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

# Save answers to file
output_file = "answersWithVectorDB.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for question in QUESTIONS:
        answer = generate_response(question)
        f.write(f"Q: {question}\nA: {answer}\n\n")
        print(f"Q: {question}\nA: {answer}\n")  # Print to console as well

print(f"\n✅ Answers saved in {output_file}")
