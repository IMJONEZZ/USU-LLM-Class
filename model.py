import torch
from torch import nn

class SimpleLanguageModel(nn.Module):
    """Minimal language model for next-token prediction."""

    def __init__(self, vocab_size: int, embed_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.embed(x)
        return self.fc(embeddings)
