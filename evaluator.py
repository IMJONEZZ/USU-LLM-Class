import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report


class ReviewDataset(Dataset):
    """Custom dataset class for video game reviews."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class Evaluator:
    """Evaluator for a BERT-based sentiment classification model."""

    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def evaluate(self, dataset):
        """Evaluates the model on a given dataset."""
        dataloader = DataLoader(dataset, batch_size=2)
        self.model.eval()

        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask, labels = (
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    batch["labels"].to(self.device),
                )
                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds)

        print(f" Accuracy: {accuracy:.4f}")
        print(" Classification Report:\n", report)

        # Logging correct & incorrect predictions
        for i, (text, pred, label) in enumerate(
            zip(dataset.texts, all_preds, all_labels)
        ):
            if pred == label:
                print(f"Correct: {text} -> Predicted: {pred}")
            else:
                print(f"Wrong: {text} -> Predicted: {pred}, Actual: {label}")

        return {
            "accuracy": accuracy,
            "report": report,
            "predictions": all_preds,
            "labels": all_labels,
        }


if __name__ == "__main__":
    # Example Validation & Test Data (Black Myth: Wukong Reviews)
    val_texts = [
        "A Souls-like experience that is challenging yet rewarding, with every level up making you feel more powerful.",
        "The combat is fluid and satisfying, requiring both strategy and skill to overcome its tough enemies.",
    ]
    val_labels = [1, 1]  # 1 = Positive

    test_texts = [
        "Boss fights feel overly punishing, with relentless attacks and limited stamina, making combat frustrating.",
        "The final boss completely undermines the journey, nullifying all your hard-earned progress.",
    ]
    test_labels = [0, 0]  # 0 = Negative

    evaluator = Evaluator()

    val_dataset = ReviewDataset(val_texts, val_labels, evaluator.tokenizer)
    test_dataset = ReviewDataset(test_texts, test_labels, evaluator.tokenizer)

    print("\n Evaluating Validation Set:")
    evaluator.evaluate(val_dataset)

    print("\n Evaluating Test Set:")
    evaluator.evaluate(test_dataset)
