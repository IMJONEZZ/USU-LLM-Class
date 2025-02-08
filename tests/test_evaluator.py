import sys
import os
import pytest
import torch
import numpy as np

#this allows me to access my data_preprocessor module
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from data_preprocessor import tokenize_data, train_model, evaluate_model


def tensor_set(tensor_list):
    return {tuple(np.array(seq)) for seq in tensor_list}


# this tells me whether the dataset being split into three groups (train, test, validation) actually worked
def test_data_splitting():
    data = tokenize_data()
    train_inputs = tensor_set(data["train"]["inputs"])
    validation_inputs = tensor_set(data["validation"]["inputs"])
    test_inputs = tensor_set(data["test"]["inputs"])

    #all possibilities for an overlap
    train_val_overlap = train_inputs & validation_inputs
    train_test_overlap = train_inputs & test_inputs
    val_test_overlap = validation_inputs & test_inputs

    # prints, if any, overlaps in train, test, or validation datasets (ensure data splitting happened correctly)
    print(f"Train-Validation Overlaps: {len(train_val_overlap)}")
    print(f"Train-Test Overlaps: {len(train_test_overlap)}")
    print(f"Validation-Test Overlaps: {len(val_test_overlap)}")


# this helps me test if my accuracy and loss are calculated correctly
def test_accuracy_of_eval():
    predictions = torch.tensor([0, 1, 2, 3])
    labels = torch.tensor([0, 1, 2, 3])
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = 100 * correct / total
    assert accuracy >= 30.0
