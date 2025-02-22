import pytest
from main import get_trainer
from transformers import TrainingArguments
from datasets import Dataset, DatasetDict

# Fake example data for testing purposes
fake_train_data = {
    "text": [
        "This is a test sentence.",
        "Another test sentence.",
        "Yet another example.",
        "One more fake sentence for the dataset.",
    ]
}

fake_eval_data = {
    "text": [
        "Fake evaluation sentence.",
        "Another fake evaluation sentence.",
    ]
}

# Create fake datasets from the fake data
train_dataset = Dataset.from_dict(fake_train_data)
eval_dataset = Dataset.from_dict(fake_eval_data)

# Wrap these datasets in a DatasetDict as it would be in a real setup
fake_datasets = DatasetDict({"train": train_dataset, "validation": eval_dataset})


# Test 1: Check if trainer is initialized correctly with fake data
def test_trainer_initialization_with_fake_data():
    # Override the main trainer with the fake dataset
    trainer = get_trainer()

    trainer.train_dataset = fake_datasets["train"]
    trainer.eval_dataset = fake_datasets["validation"]

    # Check if trainer is not None
    assert trainer is not None, "Trainer should be initialized"

    # Ensure trainer has necessary attributes
    assert hasattr(trainer, "model"), "Trainer should have 'model' attribute"
    assert hasattr(trainer, "args"), "Trainer should have 'args' attribute"
    assert hasattr(trainer, "train_dataset"), (
        "Trainer should have 'train_dataset' attribute"
    )
    assert hasattr(trainer, "eval_dataset"), (
        "Trainer should have 'eval_dataset' attribute"
    )

    # Check if the type of arguments is TrainingArguments
    assert isinstance(trainer.args, TrainingArguments), (
        "Training arguments should be an instance of TrainingArguments"
    )


# Test 2: Check if the training and evaluation datasets are correctly set
def test_datasets_with_fake_data():
    trainer = get_trainer()

    trainer.train_dataset = fake_datasets["train"]
    trainer.eval_dataset = fake_datasets["validation"]

    # Check that the trainer has both training and evaluation datasets
    assert trainer.train_dataset is not None, "Trainer should have a training dataset"
    assert trainer.eval_dataset is not None, "Trainer should have an evaluation dataset"

    # Verify that the datasets are subsets of the fake dataset
    assert len(trainer.train_dataset) == len(fake_datasets["train"]), (
        "Training dataset length mismatch"
    )
    assert len(trainer.eval_dataset) == len(fake_datasets["validation"]), (
        "Evaluation dataset length mismatch"
    )


# Test 3: Check if 'train' method exists and is callable
def test_train_method_exists():
    trainer = get_trainer()

    # Ensure 'train' method exists and is callable
    assert hasattr(trainer, "train"), "'train' method should exist in the trainer"
    assert callable(trainer.train), "'train' should be callable"


# Test 4: Check if model can be trained with the fake data (no full training, just test method)
def test_train_method_runs_with_fake_data():
    trainer = get_trainer()

    trainer.train_dataset = fake_datasets["train"]
    trainer.eval_dataset = fake_datasets["validation"]

    # Skip actual training to avoid long computation. Just check if method can run.
    try:
        trainer.train()
    except Exception as e:
        pytest.fail(f"Training failed with exception: {e}")


# Test 5: Check if the fake data fits within the Trainer's configuration
def test_fake_data_fits_in_trainer():
    trainer = get_trainer()

    # Assign fake datasets
    trainer.train_dataset = fake_datasets["train"]
    trainer.eval_dataset = fake_datasets["validation"]

    # Check if datasets are not empty
    assert len(trainer.train_dataset) > 0, "Training dataset should contain data"
    assert len(trainer.eval_dataset) > 0, "Evaluation dataset should contain data"


# Test 6: Check TrainingArguments values for the fake dataset
def test_training_arguments_with_fake_data():
    trainer = get_trainer()

    # Check specific values in the training arguments
    assert trainer.args.per_device_train_batch_size == 2, (
        "Expected train batch size to be 2"
    )
    assert trainer.args.per_device_eval_batch_size == 2, (
        "Expected eval batch size to be 2"
    )
    assert trainer.args.learning_rate == 2e-4, "Expected learning rate to be 2e-4"
    assert trainer.args.num_train_epochs == 1, "Expected number of epochs to be 1"
    assert trainer.args.logging_steps == 10, "Expected logging steps to be 10"


# Test 7: Check model's trainable parameters
def test_model_trainable_parameters():
    trainer = get_trainer()
    model = trainer.model

    # Ensure the model has trainable parameters
    assert any(p.requires_grad for p in model.parameters()), (
        "Model should have trainable parameters"
    )


# Run the tests
if __name__ == "__main__":
    pytest.main()
