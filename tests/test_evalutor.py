import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pytest

nltk.download("wordnet")
nltk.download("omw-1.4")

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


# 1️ **Define the Model**
class BERTModel(nn.Module):
    def __init__(self, hidden_size=768, vocab_size=30522):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state)
        return logits


# 2️ **Initialize Tokenizer**
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# 3️ **Dataset Class with Preprocessing**
class BERT_Dataset(Dataset):
    def __init__(self, texts, targets, tokenizer, block_size=128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text, target = self.texts[idx], self.targets[idx]

        # Tokenize input and target text
        encodings = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.block_size,
            return_tensors="pt",
        )
        target_encodings = self.tokenizer(
            target,
            truncation=True,
            padding="max_length",
            max_length=self.block_size,
            return_tensors="pt",
        )

        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "target_text": target,
            "input_text": text,
        }


# 4️ **Helper Function: Lemmatization**
def preprocess_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


# 5️ **Evaluator: TF-IDF Cosine Similarity**
def evaluate_tfidf(model, dataloader, device):
    model.eval()
    all_generated_texts, all_reference_texts = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )
            reference_texts = batch["target_text"]

            # Fake "generated" output (since BERT isn't a generative model)
            outputs = model(input_ids, attention_mask)
            generated_texts = [
                tokenizer.decode(torch.argmax(output, dim=-1), skip_special_tokens=True)
                for output in outputs
            ]

            all_generated_texts.extend(generated_texts)
            all_reference_texts.extend(reference_texts)

    # Preprocess for TF-IDF
    all_generated_texts = [preprocess_text(t) for t in all_generated_texts]
    all_reference_texts = [preprocess_text(t) for t in all_reference_texts]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_generated_texts + all_reference_texts)

    # Compute Cosine Similarity
    gen_vectors = tfidf_matrix[: len(all_generated_texts)]
    ref_vectors = tfidf_matrix[len(all_generated_texts) :]
    scores = cosine_similarity(gen_vectors, ref_vectors).diagonal()

    return scores.mean() if len(scores) > 0 else 0.0


# 6️ **Training Loop (Simplified)**
def train(model, dataloader, optimizer, loss_fn, device, num_epochs=3):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            input_ids, attention_mask = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
            )

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} Training Loss: {loss.item():.4f}")


# 7️ **Testing with Pytest**
@pytest.fixture
def sample_data():
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "AI is transforming the world.",
    ]
    targets = [
        "A fast fox leaps over a sleepy dog.",
        "Artificial intelligence changes the world.",
    ]
    return texts, targets


def test_dataset(sample_data):
    texts, targets = sample_data
    dataset = BERT_Dataset(texts, targets, tokenizer)
    assert len(dataset) == 2
    sample = dataset[0]
    assert "input_ids" in sample and "attention_mask" in sample
    assert "target_text" in sample and "input_text" in sample


def test_preprocessing():
    assert preprocess_text("running jumped") == "running jumped"
    assert preprocess_text("cats dogs") == "cat dog"  # Lemmatization check


def test_tfidf_evaluation(sample_data):
    texts, targets = sample_data
    dataset = BERT_Dataset(texts, targets, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2)

    model = BERTModel()
    score = evaluate_tfidf(model, dataloader, "cpu")
    assert 0.0 <= score <= 1.0  # Cosine similarity range


# Run Pytest with: `pytest filename.py`


# @pytest.fixture
# def sample_data():
#     """Sample dataset for testing."""
#     identical_texts = ["This is a test sentence."]
#     different_texts = ["Machine learning is fascinating."]
#     random_texts = ["Bananas are yellow."]
#     return identical_texts, different_texts, random_texts

# @pytest.fixture
# def create_dataloader():
#     """Creates a dataloader for a given set of texts and targets."""
#     def _create(texts, targets, tokenizer, batch_size=1):
#         dataset = BERT_Dataset(texts, targets, tokenizer)
#         return DataLoader(dataset, batch_size=batch_size, shuffle=False)
#     return _create

# def test_dataset_structure(sample_data, tokenizer):
#     """Test that dataset returns correctly formatted samples."""
#     texts, targets, _ = sample_data
#     dataset = BERT_Dataset(texts, targets, tokenizer)

#     assert len(dataset) == len(texts)

#     sample = dataset[0]
#     assert all(key in sample for key in ["input_ids", "attention_mask", "target_text", "input_text"]), "Dataset keys missing!"

# def test_text_preprocessing():
#     """Ensure lemmatization works correctly."""
#     assert preprocess_text("running jumped") == "running jumped"
#     assert preprocess_text("cats dogs") == "cat dog"  # Lemmatization check

# def test_tfidf_evaluation(sample_data, create_dataloader, model, device, tokenizer):
#     """Ensure TF-IDF similarity values are within a valid range."""
#     texts, targets, _ = sample_data
#     dataloader = create_dataloader(texts, targets, tokenizer)

#     score = evaluate_tfidf(model, dataloader, device)

#     assert 0.0 <= score <= 1.0, f"Similarity score out of range: {score}"

# def test_identical_inputs(sample_data, create_dataloader, model, device, tokenizer):
#     """Ensure that identical inputs yield a similarity of ~1."""
#     texts, _, _ = sample_data
#     dataloader = create_dataloader(texts, texts, tokenizer)

#     similarity = evaluate_tfidf(model, dataloader, device)

#     assert similarity == pytest.approx(1.0, abs=1e-6), f"Expected similarity of 1.0, but got {similarity}"

# def test_completely_different_inputs(sample_data, create_dataloader, model, device, tokenizer):
#     """Ensure that completely different inputs yield a similarity near 0."""
#     _, different_texts, random_texts = sample_data
#     dataloader = create_dataloader(different_texts, random_texts, tokenizer)

#     similarity = evaluate_tfidf(model, dataloader, device)

#     assert similarity < 0.1, f"Expected similarity near 0, but got {similarity}"
