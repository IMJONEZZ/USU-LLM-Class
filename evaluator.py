from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
from main import assignment_4_pipeline
from torch.utils.data import DataLoader
from zenml.client import Client

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
model.eval()


def evaluate_model(dataloader, model):
    """Evaluates model performance on the dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    total_loss = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask).logits

            # Assuming single-class problem (adjust accordingly if needed)
            labels = torch.zeros(outputs.shape[0], dtype=torch.long).to(device)
            loss = torch.nn.functional.cross_entropy(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(dataloader)


# Initialize ZenML client
client = Client()

# Fetch the latest pipeline run
pipeline = client.get_pipeline("assignment_4_pipeline")
last_run = pipeline.runs[-1]  # Get the most recent run

# Retrieve the DataLoader step output
step_name = "create_dataloader_v1"

torch.serialization.add_safe_globals([DataLoader])

dataloader_artifact = last_run.steps[step_name].outputs["output"][0].load()
print(dataloader_artifact)


# Ensure it's a DataLoader
if isinstance(dataloader_artifact, DataLoader):
    dataloader = dataloader_artifact
else:
    raise TypeError(f"Expected DataLoader but got {type(dataloader_artifact)}")

# Evaluate the model
loss = evaluate_model(dataloader, model)
print(f"Evaluation Loss: {loss}")
