import pytest
from transformers import TrainingArguments, Trainer
from unsloth import FastLanguageModel
from datasets import Dataset

# ---------------------------
# Fixtures for Dummy Components
# ---------------------------


@pytest.fixture(scope="session")
def dummy_model_and_tokenizer():
    """
    Loads a small version of the model for testing on CPU.
    For CPU tests, we disable 4-bit quantization and fast inference.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="meta-llama/meta-Llama-3.1-8B-Instruct",
        max_seq_length=128,  # Shorter sequence length for tests
        load_in_4bit=False,  # Disable 4-bit quantization for CPU
        fast_inference=False,  # Disable fast inference
        max_lora_rank=8,  # Lower LoRA rank for faster testing
        gpu_memory_utilization=0.6,
    )
    # Force the model to run on CPU.
    model.to("cpu")
    return model, tokenizer


@pytest.fixture(scope="session")
def dummy_dataset(dummy_model_and_tokenizer):
    """
    Creates a dummy dataset and tokenizes it.
    """
    _, tokenizer = dummy_model_and_tokenizer
    data = {
        "text": ["THREEPIO: I'm here to help!", "THREEPIO: What a time to be alive!"]
    }
    dataset = Dataset.from_dict(data)

    def preprocess(example):
        tokenized = tokenizer(
            example["text"], truncation=True, max_length=128, padding="max_length"
        )
        # For language modeling, labels mirror the input_ids.
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenized_dataset = dataset.map(preprocess)
    return tokenized_dataset


@pytest.fixture(scope="session")
def trainer_instance(dummy_model_and_tokenizer, dummy_dataset):
    """
    Creates a Trainer instance configured for a quick CPU run.
    """
    model, tokenizer = dummy_model_and_tokenizer
    training_args = TrainingArguments(
        output_dir="./temp_test_model",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=5,  # Only a few steps for quick testing
        learning_rate=5e-6,
        weight_decay=0.1,
        logging_steps=1,
        save_steps=5,
        fp16=False,  # Disable fp16 for CPU
        bf16=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dummy_dataset,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
    )
    return trainer


# ---------------------------
# Tests
# ---------------------------


def test_tokenizer_preprocessing(dummy_dataset):
    """
    Verify that the tokenized dataset contains the expected keys.
    """
    example = dummy_dataset[0]
    assert "input_ids" in example, "Tokenized output missing 'input_ids'"
    assert "labels" in example, "Tokenized output missing 'labels'"
    assert example["input_ids"] == example["labels"], "Labels do not match input_ids"


def test_trainer_training_step(trainer_instance):
    """
    Check that a short training run increases the global step.
    """
    initial_step = trainer_instance.state.global_step
    trainer_instance.train()
    final_step = trainer_instance.state.global_step
    assert final_step > initial_step, "Global step did not increase after training."


def test_model_inference_cpu(dummy_model_and_tokenizer):
    """
    Verify that the model's inference produces non-empty output on CPU.
    """
    model, tokenizer = dummy_model_and_tokenizer
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Speak like a wise Jedi about the Force."}],
        tokenize=False,
        add_generation_prompt=True,
    )

    # Using vLLm's sampling parameters for generation.
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=50,
    )
    output = (
        model.fast_generate(
            [prompt],
            sampling_params=sampling_params,
            lora_request=None,
        )[0]
        .outputs[0]
        .text
    )
    assert isinstance(output, str) and len(output.strip()) > 0, (
        "Model inference returned empty output."
    )


def test_model_save_and_load(tmp_path, trainer_instance, dummy_model_and_tokenizer):
    """
    Test saving the model and tokenizer to disk, then reloading them.
    """
    model, tokenizer = dummy_model_and_tokenizer
    save_dir = tmp_path / "model_save"
    save_dir_str = str(save_dir)

    # Save the model and tokenizer.
    trainer_instance.save_model(save_dir_str)
    tokenizer.save_pretrained(save_dir_str)

    # Reload the model and tokenizer.
    loaded_model, loaded_tokenizer = FastLanguageModel.from_pretrained(save_dir_str)
    loaded_model.to("cpu")
    assert loaded_model is not None, "Loaded model is None."
    assert loaded_tokenizer is not None, "Loaded tokenizer is None."

    # Verify that generation still works with the loaded model.
    prompt = loaded_tokenizer.apply_chat_template(
        [{"role": "user", "content": "Tell me something Star Wars-like."}],
        tokenize=False,
        add_generation_prompt=True,
    )

    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=50,
    )
    output = (
        loaded_model.fast_generate(
            [prompt],
            sampling_params=sampling_params,
            lora_request=None,
        )[0]
        .outputs[0]
        .text
    )
    assert isinstance(output, str) and len(output.strip()) > 0, (
        "Reloaded model inference returned empty output."
    )
