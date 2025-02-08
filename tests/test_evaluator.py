import sys
import os
import numpy as np

# this allows me to access my data_preprocessor module
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from pipeline import tokenize_data


def tensor_set(tensor_list):
    return {tuple(np.array(seq)) for seq in tensor_list}


# this tells me whether the dataset being split into three groups (train, test, validation) actually worked
def test_data_splitting():
    data = tokenize_data()
    train_inputs = tensor_set(data["train"]["inputs"])
    validation_inputs = tensor_set(data["validation"]["inputs"])
    test_inputs = tensor_set(data["test"]["inputs"])

    # all possibilities for an overlap
    train_val_overlap = train_inputs & validation_inputs
    train_test_overlap = train_inputs & test_inputs
    val_test_overlap = validation_inputs & test_inputs

    # prints, if any, overlaps in train, test, or validation datasets (ensure data splitting happened correctly)
    print(f"Train-Validation Overlaps: {len(train_val_overlap)}")
    print(f"Train-Test Overlaps: {len(train_test_overlap)}")
    print(f"Validation-Test Overlaps: {len(val_test_overlap)}")
