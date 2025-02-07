#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("omw-1.4")

# Set device (no GPU assumed)
device = "cpu"


# In[ ]:


# Step 1: Preprocessing - Lemmatization
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)


# Step 2: Define the Model Class
class BERT_LLM(nn.Module):
    def __init__(self, hidden_size=768, vocab_size=30522):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state)
        return logits


# Step 3: Initialize Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Step 4: Custom Dataset Class
class BERT_Dataset(Dataset):
    def __init__(self, texts, targets, tokenizer, block_size=128):
        self.texts = [preprocess_text(t) for t in texts]
        self.targets = [preprocess_text(t) for t in targets]
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text, target = self.texts[idx], self.targets[idx]
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


# In[ ]:


# Step 5: Create Dataloaders
train_texts = ["The cat sat on the mat.", "Dogs are loyal animals."]
train_targets = ["A cat was sitting on a rug.", "Loyal animals include dogs."]

val_texts = ["Birds can fly.", "Fish live in water."]
val_targets = ["Flying is natural for birds.", "Water is the habitat of fish."]

test_texts = ["New input sentence.", "Another example for testing."]
test_targets = ["Expected output sentence.", "Ground truth for testing."]


# #### more text
#

# In[ ]:


train_texts += [
    "She reads books every night.",
    "The cat sat on the mat.",
    "We went to the park yesterday.",
    "He drinks coffee in the morning.",
    "They are listening to music.",
    "The flowers are blooming.",
    "I like to bake cookies.",
    "She sings beautifully.",
    "He plays the guitar.",
    "We watch movies on weekends.",
    "They went swimming in the ocean.",
    "The sun is shining brightly.",
    "I am learning Spanish.",
    "She works as a doctor.",
    "He drives a car.",
    "We live in a big city.",
    "They travel around the world.",
    "The dog barks loudly.",
    "I enjoy hiking in the mountains.",
    "She paints beautiful pictures.",
    "He writes poetry.",
    "We celebrate birthdays with cake.",
    "They play soccer in the field.",
    "The rain is falling gently.",
    "I love to eat pizza.",
    "She dances gracefully.",
    "He tells funny jokes.",
    "We visit our grandparents often.",
    "They study hard for exams.",
    "The stars twinkle at night.",
    "I am excited about the future.",
    "She believes in herself.",
    "He is a kind and generous person.",
    "We are grateful for our friends and family.",
    "They are making a difference in the world.",
    "The clock is ticking.",
    "I need to buy groceries.",
    "She has a new job.",
    "He is going on vacation.",
    "We are planning a party.",
    "They are building a house.",
    "The leaves are changing color.",
    "I am feeling happy today.",
    "She is wearing a beautiful dress.",
    "He is reading an interesting book.",
    "We are having a picnic.",
    "They are learning a new language.",
    "The moon is full tonight.",
    "I am tired of working.",
    "She is looking for a new apartment.",
    "He is thinking about the future.",
    "We are going to the concert.",
    "They are celebrating their anniversary.",
    "The snow is falling softly.",
    "I am dreaming of a white Christmas.",
    "She is decorating the tree.",
    "He is wrapping presents.",
    "We are singing Christmas carols.",
    "They are enjoying the holiday season.",
    "The fire is burning brightly.",
    "I am staying warm by the fire.",
    "She is drinking hot chocolate.",
    "He is telling Christmas stories.",
    "We are playing board games.",
    "They are having a wonderful time.",
]

train_targets += [
    "She spends her evenings reading.",
    "The feline rested on the rug.",
    "We visited the park the previous day.",
    "He consumes coffee each morning.",
    "They are currently listening to music.",
    "The flowers are in full bloom.",
    "I enjoy baking cookies.",
    "She has a beautiful singing voice.",
    "He is a guitarist.",
    "We enjoy watching movies during weekends.",
    "They went swimming in the ocean.",
    "The sun is shining brightly.",
    "I am currently studying Spanish.",
    "She is a medical professional.",
    "He is a driver.",
    "We reside in a large city.",
    "They travel internationally.",
    "The canine barks loudly.",
    "I find hiking in the mountains enjoyable.",
    "She creates beautiful paintings.",
    "He is a poet.",
    "We celebrate birthdays with cake.",
    "They play soccer on the field.",
    "Rain is falling gently.",
    "I love eating pizza.",
    "She is a graceful dancer.",
    "He is a comedian.",
    "We frequently visit our grandparents.",
    "They study diligently for exams.",
    "The stars twinkle at night.",
    "I am optimistic about the future.",
    "She has self-belief.",
    "He is a compassionate and generous individual.",
    "We are thankful for our friends and family.",
    "They are contributing to positive change in the world.",
    "Time is passing.",
    "I need to purchase groceries.",
    "She has started a new job.",
    "He is taking a vacation.",
    "We are organizing a party.",
    "They are constructing a house.",
    "The leaves are changing color.",
    "I am feeling happy today.",
    "She is wearing an elegant dress.",
    "He is engrossed in an interesting book.",
    "We are having a picnic.",
    "They are learning a new language.",
    "The moon is full tonight.",
    "I am tired of working.",
    "She is searching for a new apartment.",
    "He is contemplating the future.",
    "We are attending the concert.",
    "They are celebrating their anniversary.",
    "Snow is falling softly.",
    "I am dreaming of a white Christmas.",
    "She is decorating the tree.",
    "He is wrapping presents.",
    "We are singing Christmas carols.",
    "They are enjoying the holiday season.",
    "The fire is burning brightly.",
    "I am staying warm by the fire.",
    "She is drinking hot chocolate.",
    "He is telling Christmas stories.",
    "We are playing board games.",
    "They are having a wonderful time.",
]


val_texts += [
    "The sun sets in the west.",
    "She enjoys reading science fiction.",
    "He is a talented musician.",
    "They are planning a trip to Europe.",
    "The rain is pouring down.",
]
val_targets += [
    "The sun goes down in the west.",
    "She likes to read science fiction books.",
    "He is a skilled musician.",
    "They are going on a trip to Europe.",
    "It is raining heavily.",
]

test_texts += [
    "The wind is blowing strongly.",
    "He likes to play video games.",
    "She is a successful businesswoman.",
    "They are having a barbecue.",
    "The flowers smell beautiful.",
]
test_targets += [
    "The wind is blowing hard.",
    "He enjoys playing video games.",
    "She is a prosperous businesswoman.",
    "They are having a cookout.",
    "The flowers have a lovely fragrance.",
]


# #### continuing

# In[ ]:


train_dataset = BERT_Dataset(train_texts, train_targets, tokenizer)
val_dataset = BERT_Dataset(val_texts, val_targets, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Step 6: Training Setup
model = BERT_LLM().to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.CrossEntropyLoss()

# Step 7: Training Loop
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(
            outputs.view(-1, model.classifier.out_features), input_ids.view(-1)
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_dataloader):.4f}")

    # Validation
    print("\nValidation Evaluation:")
    val_score = evaluate_tfidf(model, val_dataloader, device)
    print(f"TF-IDF Cosine Similarity Score: {val_score:.4f}")

# Step 8: Save Model
torch.save(model.state_dict(), "bert_llm.pth")


# Step 9: Define TF-IDF Evaluation
def evaluate_tfidf(model, dataloader, device):
    model.eval()
    all_generated_texts = []
    all_reference_texts = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask)

        # Convert model outputs to text (dummy approach: use original input)
        generated_texts = batch["input_text"]
        all_generated_texts.extend(generated_texts)
        all_reference_texts.extend(batch["target_text"])

    # Compute TF-IDF Cosine Similarity
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_generated_texts + all_reference_texts)

    num_samples = len(all_generated_texts)
    generated_tfidf = tfidf_matrix[:num_samples]
    reference_tfidf = tfidf_matrix[num_samples:]

    scores = cosine_similarity(generated_tfidf, reference_tfidf).diagonal()
    return np.mean(scores)


# Step 10: Final Testing
model.load_state_dict(torch.load("bert_llm.pth"))
print("\nFinal Test Evaluation:")
test_score = evaluate_tfidf(model, val_dataloader, device)
print(f"Final TF-IDF Cosine Similarity Score: {test_score:.4f}")


# In[ ]:


test_dataset = BERT_Dataset(test_texts, test_targets, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# Final test evaluation
print("\nFinal Test Evaluation:")
test_score = evaluate_tfidf(model, test_dataloader, device)
print(f"Final TF-IDF Cosine Similarity Score: {test_score:.4f}")
