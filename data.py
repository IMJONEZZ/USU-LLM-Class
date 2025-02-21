from datasets import load_dataset
from config import DATASET_NAME


def format_instruction(sample):
    return {
        "text": f"""<|im_start|>system
        You are a Cypher query expert. Generate valid Cypher queries based on the input question and database schema.
        Schema: {sample["schema"]}<|im_end|>
        <|im_start|>user
        {sample["question"]}<|im_end|>
        <|im_start|>assistant
        {sample["cypher"]}"""
    }


def load_and_preprocess_data(download_mode="force_redownload"):
    dataset = load_dataset(DATASET_NAME, download_mode=download_mode)
    return dataset.map(format_instruction, batched=False)


def tokenize_dataset(dataset, tokenizer, max_length=512):
    return dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
