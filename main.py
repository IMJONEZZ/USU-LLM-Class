import json
import re
from zenml.pipelines import pipeline
from zenml.steps import step


class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        text = text.lower()
        contractions = {
            "'m": "am",
            "n't": "not",
            "'re": "are",
            "'ve": "have",
            "'ll": "will",
        }
        for contraction, full in contractions.items():
            text = text.replace(contraction, full)

        preprocessed = re.split(r"([,.;:?!'\"()\\]|--|\s)", text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]

        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r"\s+([,.;:?!'\"()\\])", r"\1", text)
        return text


@step
def load_data():
    with open("SW_EpisodeIV_VI.json", "r") as file:
        data = json.load(file)

    features = [entry["Line"] for entry in data]
    labels = [entry["Character"] for entry in data]

    return {"features": features, "labels": labels}


@step
def tokenize_lines(data):
    all_lines = data["features"]
    all_tokens = sorted(list(set(re.findall(r"\w+|[^\w\s]", " ".join(all_lines)))))
    all_tokens.extend(["<endoftext>", "<|unk|>"])

    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    tokenizer = SimpleTokenizer(vocab)

    tokenized_data = [tokenizer.encode(line) for line in data["features"]]

    return {"features": tokenized_data, "labels": data["labels"]}


@step
def train_model(data, original) -> None:
    print(original["features"][0])
    print(data["features"][0])

    print(f"Trained model using {len(data['features'])} data points. ")


@pipeline
def simple_ml_pipeline():
    dataset = load_data()
    tokenized = tokenize_lines(dataset)
    train_model(tokenized, dataset)


if __name__ == "__main__":
    run = simple_ml_pipeline()
