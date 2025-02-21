from process_data import process_data
from train_model import train_model_
from evaluator import evaluate_model_, predict
from tokenizer import ImprovedTokenizer
import torch
from nn import SimpleTextClassifier  # Ensure this matches your actual model definition

# Load the dataset
data = process_data()

# Train and get model path
model_path = train_model_(data)

# Load the trained model
checkpoint = torch.load(model_path, map_location="cpu")

# Recreate the model architecture (make sure vocab_size, embed_dim, and num_classes match!)
vocab_size = 734
embed_dim = 64
num_classes = 22
model = SimpleTextClassifier(vocab_size, embed_dim, num_classes)

# Load the trained weights
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

vocab = data["vocab"]
special_characters = data["special_tokens"]

tokenizer = ImprovedTokenizer(vocab, special_characters, max_length=24)

# Test a sample input
sample_text = input()
predicted_label = predict(model, tokenizer, sample_text)

label_to_int = data["label_to_int"]


def decode_label(label_id, label_to_int):
    # Reverse the mapping
    int_to_label = {v: k for k, v in label_to_int.items()}

    # Return the corresponding label string
    return int_to_label.get(label_id, "Unknown Label")  # Default if ID is not found
evaluate_model_(data, model_path)
print(f"Predicted Label: {decode_label(predicted_label, label_to_int)}")
