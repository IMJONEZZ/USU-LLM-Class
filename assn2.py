
from zenml import pipeline, step
import re
import string
import polars as pl


@step
# Tokenize
def make_tokenizer(doc_text):
    return SimpleTokenizer(make_corpus(doc_text))


class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
                
        text = expand_contractions(text) # fix some common contractions
        text = text.lower() # lower case everything

        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text



def make_corpus(full_document_text):
    escaped_punctuation = re.escape(string.punctuation)


    full_document_text = expand_contractions(full_document_text) # fix some common contractions
    full_document_text = full_document_text.lower() # lower case everything

    preprocessed = re.split(
        fr'([{escaped_punctuation}]|--|\s)', full_document_text
    )
    preprocessed = [
        item.strip() for item in preprocessed if item.strip()
    ]
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    corpus = {token: integer for integer, token in enumerate(all_tokens)}
    # print(corpus)
    # print(len(corpus.items()))

    return corpus



@step 
def load_data(file_path):
    # print("Load the Star Wars script")
    df = pl.read_json(file_path)

    # Add <|endoftext|> to each line and create a new column Line2
    df = df.with_columns(
        pl.concat_str(
            [
                pl.col("Line"), 
                pl.lit("<|endofline|>")
            ],
            separator=" ",
        ).alias("Line2")
    )
    giant_string =  " ".join(df.select(pl.col("Line2")).to_series().to_list())
    
    return giant_string

def expand_contractions(text):
    """Expands contractions in the given text."""
    for contraction, expanded in CONTRACTIONS.items():
        text = re.sub(rf'\b{re.escape(contraction)}\b', expanded, text)
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



@pipeline
def Assn2_pipeline():
    file_path = "SW_EpisodeIV_VI.json"
    dataset = load_data(file_path)
    # corpus = make_corpus(dataset)

    # print(type(corpus))
    # print(type(corpus))
    # tokenizer = SimpleTokenizer(corpus)  # Pass the StepArtifact directly

    # print(type(tokenizer))

    make_tokenizer(dataset)
    # tokenized_output = tokenizer.encode(dataset)  # Tokenize the dataset
    
    # return tokenized_output  # Return tokenized output



if __name__ == "__main__":

   assn2 = Assn2_pipeline()

   # You can now use the `run` object to see steps, outputs, etc.
