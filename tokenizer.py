import re
import string
import polars as pl


class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        text = expand_contractions(text)  # fix common contractions
        text = text.lower()  # lower case everything

        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
        return text


# Define the SimpleTokenizer class and related functions
def expand_contractions(text):
    """Expands contractions in the given text."""
    for contraction, expanded in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(contraction)}\b", expanded, text)
    return text


CONTRACTIONS = {
    "can't": "can not",
    "won't": "will not",
    "n't": " not",
    "'ll": " will",
    "'re": " are",
    "'ve": " have",
    "'d": " would",
    "I'm": "I am",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "they're": "they are",
    "they've": "they have",
    "we're": "we are",
    "we've": "we have",
    # Add more contractions as needed
}


def make_tokenizer(doc_text):
    return SimpleTokenizer(make_corpus(doc_text))


def make_corpus(full_document_text):
    escaped_punctuation = re.escape(string.punctuation)

    full_document_text = expand_contractions(
        full_document_text
    )  # fix common contractions
    full_document_text = full_document_text.lower()  # lower case everything

    preprocessed = re.split(rf"([{escaped_punctuation}]|--|\s)", full_document_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    corpus = {token: integer for integer, token in enumerate(all_tokens)}
    return corpus


def get_string(file_path):
    # Load the data from the provided file
    df = pl.read_json(file_path)

    # Add <|endofline|> to each line and create a new column
    df = df.with_columns(
        pl.concat_str([pl.col("Line"), pl.lit("<|endofline|>")], separator=" ").alias(
            "Line2"
        )
    )
    giant_string = " ".join(df.select(pl.col("Line2")).to_series().to_list())

    return giant_string

    # Create and return the DataLoader
    # return create_dataloader_v1(giant_string, tokenizer)
