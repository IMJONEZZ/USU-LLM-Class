from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from uuid import uuid4


class WikiDataIngestion:
    def __init__(
        self,
        index,
        wikidata=None,
        embedder=None,
        tokenizer=None,
        text_splitter=None,
        batch_limit=100,
    ):
        self.index = index
        self.wikidata = wikidata or load_dataset(
            "wikipedia",
            "20220301.simple",
            split="train[:10000]",
            trust_remote_code=True,
        )
        self.embedder = embedder or SentenceTransformer("all-MiniLM-L6-v2")
        self.tokenizer = tokenizer or self.embedder.tokenizer
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=self.token_length,
            separators=["\n\n", "\n", " ", ""],
        )
        self.batch_limit = batch_limit

    def token_length(self, text):
        tokens = self.tokenizer.encode(text)
        return len(tokens)

    def get_wiki_metadata(self, page):
        return {
            "wiki-id": str(page["id"]),
            "source": page["url"],
            "title": page["title"],
        }

    def split_texts_and_metadatas(self, page):
        basic_metadata = self.get_wiki_metadata(page)
        texts = self.text_splitter.split_text(page["text"])
        metadatas = [
            {"chunk": j, "text": text, **basic_metadata} for j, text in enumerate(texts)
        ]
        return texts, metadatas

    def upload_batch(self, texts, metadatas):
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeddings = self.embedder.encode(texts)
        self.index.upsert(vectors=zip(ids, embeddings, metadatas))

    def batch_upload(self):
        batch_texts = []
        batch_metadatas = []

        for page in tqdm(self.wikidata):
            texts, metadatas = self.split_texts_and_metadatas(page)

            batch_texts.extend(texts)
            batch_metadatas.extend(metadatas)

            if len(batch_texts) >= self.batch_limit:
                self.upload_batch(batch_texts, batch_metadatas)
                batch_texts = []
                batch_metadatas = []

        if len(batch_texts) > 0:
            self.upload_batch(batch_texts, batch_metadatas)
