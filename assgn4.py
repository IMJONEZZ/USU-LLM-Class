import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.stem import WordNetLemmatizer
from rouge_score import rouge_scorer
import nltk

# Ensure necessary NLTK resources are downloaded
nltk.download("wordnet")

# Initialize tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Example data
train_texts = ["The river rushes through the valley."]
train_targets = ["The river flows swiftly through the valley."]
labels = torch.tensor([1])  # Dummy label

# Tokenization
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt"
)
target_encodings = tokenizer(
    train_targets, truncation=True, padding=True, max_length=512, return_tensors="pt"
)

# Optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Training loop (1 epoch for simplicity)
model.train()
for epoch in range(1):
    optimizer.zero_grad()

    outputs = model(**train_encodings)
    loss = criterion(outputs.logits, labels)  # Ensure correct shape

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Switch to evaluation mode
model.eval()

# Test prediction
test_text = ["The river rushes through the valley."]
test_encodings = tokenizer(
    test_text, truncation=True, padding=True, max_length=512, return_tensors="pt"
)

with torch.no_grad():
    output = model(**test_encodings)
    predicted_label = torch.argmax(output.logits, dim=-1).item()

# Reference for evaluation
reference = "The river flows swiftly through the valley."
predicted_sentence = train_texts[0]  # Using input since it's a classification task

# Lemmatization function
lemmatizer = WordNetLemmatizer()


def lemmatize_sentence(sentence):
    return " ".join([lemmatizer.lemmatize(word) for word in sentence.split()])


# Apply lemmatization
reference_lem = lemmatize_sentence(reference)
predicted_lem = lemmatize_sentence(predicted_sentence)

# ROUGE scoring
rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
scores = rouge_scorer.score(predicted_lem, reference_lem)

# Display results
print("Predicted Label:", predicted_label)
print("Reference (Lemmatized):", reference_lem)
print("Prediction (Lemmatized):", predicted_lem)
print("ROUGE Scores:", scores)
