from zenml import pipeline, step
from typing import List
import re
import json
from BPETokenizer import BPETokenizer

@step
def load_corpus(file_path: str) -> List[str]:
    """Loads the text corpus for training the tokenizer."""
    with open(file_path, "r") as file:
        data = json.load(file)

    lines = []
    for item in data:
        line = item.get("Line", "")
        lines.append(line)
    preprocessed = " ".join(lines)
    words = re.split(r'([,.:;?_!"()]|--|\s)', preprocessed)
    corpus = [t.strip() for t in words if t.strip()]
    return corpus

@step
def train_tokenizer(corpus: List[str]) -> BPETokenizer:
    """Trains the BPETokenizer on the provided corpus."""
    tokenizer = BPETokenizer(num_merges=1000)
    tokenizer.fit(corpus)
    return tokenizer

@pipeline
def text_processing_pipeline():
    """Defines a pipeline that connects the steps."""
    file_path = "C:/Users/tjker/Desktop/School/Spring_2025/USU-LLM-Class/usu_llm_class/SW_EpisodeIV_VI.json" # Hard coded for now.
    corpus = load_corpus(file_path)
    tokenizer = train_tokenizer(corpus)
    

if __name__ == "__main__":
    run = text_processing_pipeline()