import re
from collections import defaultdict, Counter

# Tokenize
class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [ item.strip() for item in preprocessed if item.strip() ]

        preprocessed = [item if item in self.str_to_int
        else "<|unk|>" for item in preprocessed]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

class WordPieceTokenizer:
    def __init__(self, vocab_size: int, corpus: list):
        self.vocab_size = vocab_size
        self.str_to_int = self.build_vocab(corpus)
        print(self.str_to_int)
        # self.int_to_str = { i:s for s,i in self.str_to_int.items()}

    def encode(self, text):
        text = text.lower()
        subword_to_id = {subword: idx for idx, subword in enumerate(self.str_to_int, start=1)}
        vocab = sorted(self.str_to_int, key=len, reverse=True)  # Sort vocab by length for longest match
        encoded_sentence = []
        
        for word in text.split():
            while word:
                match = None
                # Try to match the longest subword in the vocab
                for subword in vocab:
                    if word.startswith(subword.replace("##", "")):
                        match = subword
                        break
                if match:
                    encoded_sentence.append(subword_to_id[match])
                    word = word[len(match.replace("##", "")):]  # Remove the matched portion
                else:
                    encoded_sentence.append(subword_to_id["<|unk|>"])
                    break
        
        return encoded_sentence, subword_to_id

    def decode(self, ids):
        tokens = [self.int_to_str[i] for i in ids]
        text = " ".join(tokens)
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
    def build_vocab(self, corpus: list) -> set:
        vocab = set(f'##{char}' for word in corpus for char in word)
        vocab_count = {char: count for char, count in Counter("".join(corpus)).items()}
        tokenized_corpus = [[f"##{char}" for char in word] for word in corpus]
        
        while len(vocab) < self.vocab_size:
            # Step 2: Find most frequent subword pair
            scores = self.get_pair_scores(tokenized_corpus, vocab_count)
            if not scores:
                break
            
            # Find the best pair to merge
            best_pair = max(scores, key=scores.get)
            new_token = best_pair[0] + best_pair[1].replace("##", "")
            
            # Update the tokenized corpus
            for i, tokens in enumerate(tokenized_corpus):
                new_tokens = []
                skip = False
                for j in range(len(tokens)):
                    if skip:
                        skip = False
                        continue
                    if j < len(tokens) - 1 and (tokens[j], tokens[j + 1]) == best_pair:
                        new_tokens.append(new_token)
                        skip = True
                    else:
                        new_tokens.append(tokens[j])
                tokenized_corpus[i] = new_tokens
            
            # Update vocab and counts
            vocab.add(new_token)
            vocab_count[new_token.replace('#', '')] = sum(new_token in word for word in tokenized_corpus)
            for token in best_pair:
                vocab_count[token.replace('#', '')] -= vocab_count[new_token.replace('#', '')]

        vocab.add("<|unk|>")
        return vocab
    
    def get_pair_scores(self, tokenized_corpus, vocab_count):
        pair_counts = defaultdict(int)
        for tokens in tokenized_corpus:
            for i in range(len(tokens) - 1):
                pair_counts[(tokens[i], tokens[i + 1])] += 1
        
        # Calculate scores
        scores = {}
        for pair, freq in pair_counts.items():
            score = freq / (vocab_count[pair[0].replace('#', '')] * vocab_count[pair[1].replace('#', '')])
            scores[pair] = score
        return scores
    
if __name__ == "__main__":
    print("Wrong file")
    # tokenizer = WordPieceTokenizer(vocab)
    # text = "Hello, World."
    # ids = tokenizer.encode(text)
    # print(ids)
    # text = tokenizer.decode(ids)
    # print(text)
    # print(tokenizer.int_to_str.keys())
    # print(tokenizer.int_to_str.values())