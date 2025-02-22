import sys
import os
import hashlib
import torch
import numpy as np
import random

module_path = os.path.abspath("../src")
if module_path not in sys.path:
    sys.path.append(module_path)

try:
    from train_model import train_model_
    from process_data import process_data
except ImportError as e:
    print(f"Error importing modules: {e}")


def test_model_saved():
    model_path = "best_model.pth"

    # Ensure the file exists before training
    if os.path.exists(model_path):
        before_mod_time = os.path.getmtime(model_path)
    else:
        before_mod_time = None  # Model does not exist initially

    train_model_(process_data())  # Train and save model

    assert os.path.exists(model_path), "Model file was not saved!"

    after_mod_time = os.path.getmtime(model_path)

    # If the model existed before, check that it was updated
    if before_mod_time is not None:
        assert after_mod_time > before_mod_time, "Model file was not updated"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)  # Call before training & inference


def get_file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def test_model_consistency():
    model_path = "best_model.pth"

    # train and save model
    train_model_(process_data())

    # ensure model is saved
    assert os.path.exists(model_path), "Model file was not saved"

    # get the first hash
    first_hash = get_file_hash(model_path)

    # train again
    train_model_(process_data())

    # get second hash
    second_hash = get_file_hash(model_path)

    # ensure the model file is identical acrosss runs
    assert first_hash == second_hash, "Model file is inconsistent across runs"
