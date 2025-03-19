def test_dataset_loading(file_path):
    """Ensure dataset loads correctly from JSON file"""
    dataset = load_dataset("json", data_files={"train": file_path})["train"]
    assert len(dataset) > 0, "Dataset is empty"
    assert "Character" in dataset.column_names, "Missing 'Character' column"
    assert "Line" in dataset.column_names, "Missing 'Line' column"

    print("✅ Dataset loading test passed")


# Call this function with your dataset path
test_dataset_loading("/kaggle/input/star-wars3/star-wars.json")


def test_tokenization(tokenizer, sample_text="Yoda: Do or do not, there is no try."):
    """Ensure the tokenizer processes text correctly"""
    tokens = tokenizer(
        sample_text, truncation=True, max_length=512, padding="max_length"
    )
    assert "input_ids" in tokens, "Missing 'input_ids' key"
    assert len(tokens["input_ids"]) == 512, "Tokenized output length mismatch"

    print("✅ Tokenizer test passed")


# Call with your tokenizer
test_tokenization(tokenizer)


def test_preprocessing(dataset, tokenizer):
    """Ensure preprocessing function correctly tokenizes dataset"""
    processed_data = dataset.map(lambda x: preprocess(x), batched=False)
    sample = processed_data[0]

    assert "input_ids" in sample, "Tokenized sample missing 'input_ids'"
    assert len(sample["input_ids"]) == max_seq_length, (
        "Tokenized sequence length mismatch"
    )
    assert sample["input_ids"] == sample["labels"], (
        "Mismatch between input_ids and labels"
    )

    print("✅ Data preprocessing test passed")


# Call with dataset and tokenizer
test_preprocessing(dataset, tokenizer)


def test_model_forward(model, tokenizer, sample_text="Tell me a story about Yoda"):
    """Ensure the model performs a forward pass correctly"""
    tokens = tokenizer(sample_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)

    assert hasattr(outputs, "loss") or hasattr(outputs, "logits"), (
        "Model output missing expected attributes"
    )

    print("✅ Model forward pass test passed")


# Call with model and tokenizer
test_model_forward(model, tokenizer)


def test_training_step(trainer):
    """Ensure the training step runs without errors"""
    try:
        trainer.train(resume_from_checkpoint=False)
        print("✅ Training step test passed")
    except Exception as e:
        print(f"❌ Training step test failed: {e}")


# Call with trainer
test_training_step(trainer)


from transformers import AutoModel


def test_model_saving_loading(model, save_path="/kaggle/working/test_model"):
    """Ensure the model saves and reloads correctly"""
    model.save_pretrained(save_path)
    assert os.path.exists(save_path), "Model save directory not found"

    loaded_model = AutoModel.from_pretrained(save_path)
    assert loaded_model is not None, "Failed to reload saved model"

    print("✅ Model saving and loading test passed")


# Call with your model
test_model_saving_loading(model)


def test_generation(model, tokenizer):
    """Ensure the model generates coherent text"""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Tell me a bedtime story with Yoda's voice"}],
        tokenize=False,
        add_generation_prompt=True,
    )

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=100,
    )

    output = (
        model.fast_generate([text], sampling_params=sampling_params, lora_request=None)[
            0
        ]
        .outputs[0]
        .text
    )

    assert isinstance(output, str), "Generated output is not a string"
    assert len(output) > 10, "Generated output is too short"

    print("✅ Model generation test passed")


# Call with model and tokenizer
test_generation(model, tokenizer)
