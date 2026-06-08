## Code: Embedding with Scale

import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq_len)  →  output: (batch, seq_len, d_model)
        return self.embed(token_ids) * math.sqrt(self.d_model)


# Quick check
vocab_size, d_model = 30522, 512
embed = TokenEmbedding(vocab_size, d_model)
ids = torch.randint(0, vocab_size, (2, 10))   # batch=2, seq=10
out = embed(ids)
print(out.shape)  # torch.Size([2, 10, 512])

## Demo: Sentence Transformer Embedding (Pretrained)

# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Machine learning is powerful",
    "Deep learning uses neural networks",
    "The cat sat on the mat",
]

embeddings = model.encode(sentences)
print(f"Shape: {embeddings.shape}")   # (3, 384)

# Cosine similarity between first two (related) vs first and third (unrelated)
from numpy.linalg import norm
cos = lambda a, b: (a @ b) / (norm(a) * norm(b))
print(f"ML vs DL similarity:  {cos(embeddings[0], embeddings[1]):.4f}")
print(f"ML vs cat similarity: {cos(embeddings[0], embeddings[2]):.4f}")
