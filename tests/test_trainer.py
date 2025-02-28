import pytest
import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

# Conditional import for model loading
if torch.cuda.is_available():
    from unsloth import FastLanguageModel
else:
    from transformers import AutoModel as FastLanguageModel


# ✅ Fixture for loading model and tokenizer once per test session
@pytest.fixture(scope="session")
def model_and_tokenizer():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        max_seq_length=512,
        load_in_4bit=True if torch.cuda.is_available() else False,
    )
    return model, tokenizer


# ✅ Fixture for loading Wikipedia dataset
@pytest.fixture(scope="session")
def wiki_dataset():
    dataset = load_dataset("wikipedia", "20220301.simple")

    def format_wiki_as_prompt(example):
        example["prompt"] = f"Tell me about this topic:\n\n{example['text']}"
        return example

    return dataset.map(format_wiki_as_prompt, remove_columns=["text"])


# ✅ Fixture for initializing training arguments
@pytest.fixture(scope="session")
def training_args():
    return GRPOConfig(
        learning_rate=5e-6,
        max_steps=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        output_dir="outputs",
    )


# ✅ Fixture for initializing trainer
@pytest.fixture(scope="session")
def trainer(model_and_tokenizer, training_args):
    model, tokenizer = model_and_tokenizer
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
    )


# ✅ Fixture for setting sampling parameters
@pytest.fixture(scope="session")
def sampling_params():
    return SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=1024,
    )


def test_model_loading(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    assert model is not None
    assert tokenizer is not None


def test_tokenization(model_and_tokenizer):
    _, tokenizer = model_and_tokenizer
    sample_text = "This is a test sentence."
    tokenized = tokenizer(sample_text, truncation=True, padding=True, max_length=512)
    assert "input_ids" in tokenized
    assert len(tokenized["input_ids"]) <= 512


def test_dataset_loading(wiki_dataset):
    assert wiki_dataset is not None
    assert "prompt" in wiki_dataset["train"].column_names


def test_training_initialization(trainer):
    assert trainer is not None


def test_training_execution(trainer):
    trainer.train()
    assert trainer.state.global_step == 2


def test_inference(model_and_tokenizer, sampling_params):
    model, tokenizer = model_and_tokenizer
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is the Riddle of the Sphinx?"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    output = (
        model.fast_generate([text], sampling_params=sampling_params)[0].outputs[0].text
    )
    assert isinstance(output, str)
    assert len(output) > 0
