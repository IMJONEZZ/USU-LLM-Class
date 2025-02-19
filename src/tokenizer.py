import re


class ImprovedTokenizer:
    # Tokenizer with preprocessing for lowercasing and contraction handling

    def __init__(
        self, vocab: dict, special_tokens: dict | None = None, max_length: int = 24
    ):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
        self.special_tokens = special_tokens or {}
        self._add_special_tokens(self.special_tokens)
        self.max_length = max_length  # Sets max sequence length

    def _add_special_tokens(self, tokens: dict) -> None:
        for token_symbol in tokens.values():
            if token_symbol not in self.str_to_int:
                new_id = len(self.str_to_int)  # Assign next available ID
                self.str_to_int[token_symbol] = new_id
                self.int_to_str[new_id] = token_symbol

    def add_special_tokens(self, tokens: dict) -> None:
        # Public method to add special tokens dynamically.
        self._add_special_tokens(tokens)

    def preprocess(self, text: str) -> str:
        # Lowercase text and expand common contractions
        text = text.lower()
        contraction_dict = {
            "m": "am",
            "s": "is",
            "t": "not",
            "re": "are",
            "ve": "have",
            "ll": "will",
            "d": "would",
            "N": "not",
            "n": "not",
            "c": "can",
        }

        def replace_contractions(match: re.Match) -> str:
            word, contraction = match.group(1), match.group(2)
            return f"{word} {contraction_dict.get(contraction, contraction)}"

        return re.sub(
            r"([a-zA-Z])'(m|s|t|re|ve|ll|d|N|n|c)(?=\s|$)", replace_contractions, text
        )

    def encode(self, text: str) -> list[int]:
        # Encode text into token IDs
        text = self.preprocess(text)
        tokens = [
            token.strip()
            for token in re.split(r"([,.:;?_!\"()']|--|\s)", text)
            if token.strip()
        ]
        tokens = [
            token
            if token in self.str_to_int
            else self.special_tokens.get("unk_token", "<|unk|>")
            for token in tokens
        ]
        token_ids = [
            self.str_to_int.get(token, self.str_to_int["<|unk|>"]) for token in tokens
        ]

        # Pad or truncate the sequence
        return token_ids[: self.max_length] + [
            self.str_to_int.get(self.special_tokens["pad_token"], 0)
        ] * max(0, self.max_length - len(token_ids))

    def decode(self, ids: list[int]) -> str:
        # Decode token IDs back into text
        text = " ".join(self.int_to_str[i] for i in ids)
        return re.sub(r"\s+([,.:;?\"()'])", r"\1", text)
