import torch
import pytest
from datasets import load_dataset
from transformers import BertTokenizer
import evaluate


@pytest.fixture
def sample_data():
    dataset = load_dataset("neo4j/text2cypher-2024v1")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["question"], truncation=True, max_length=128, padding="max_length"
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["cypher"],
                truncation=True,
                max_length=128,
                padding="max_length",
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, remove_columns=dataset["test"].column_names
    )

    tokenized_datasets.set_format(type="torch")
    subset_val = tokenized_datasets["test"].select(range(1))
    return subset_val


def test_bleu_evaluation(sample_data):
    model = torch.load("model.pt", weights_only=False)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bleu_metric = evaluate.load("bleu")

    output_ids = model.generate(
        sample_data[0]["input_ids"].unsqueeze(0),
        max_new_tokens=50,
        num_beams=4,
        early_stopping=True,
    )

    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    bleu_metric = evaluate.load("bleu")
    ref = tokenizer.decode(sample_data[0]["labels"], skip_special_tokens=True)

    predictions = [prediction]
    references = [[ref]]
    result = bleu_metric.compute(
        predictions=predictions, references=references, smooth=True
    )
    assert 0.0 <= result["bleu"]
