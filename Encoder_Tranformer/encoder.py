"""
Complete Encoder Transformer implementation in PyTorch.
Covers: ScaledDotProductAttention, MultiHeadAttention,
        PositionwiseFFN, EncoderLayer, full Encoder stack,
        padding mask utility, classification head example.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────────────────────────
# 1.  SCALED DOT-PRODUCT ATTENTION
# ─────────────────────────────────────────────────────────────────

class ScaledDotProductAttention(nn.Module):
    """
    Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

    Supports both padding masks and causal (look-ahead) masks.
    mask values: 0 = masked (set to -inf), 1 = keep
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        d_k = Q.size(-1)

        # Raw scores: (B, heads, seq_q, seq_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask=0 → -inf → softmax gives 0 weight
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)   # (B, heads, seq_q, seq_k)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, V)     # (B, heads, seq_q, d_v)
        return output, attn_weights


# ─────────────────────────────────────────────────────────────────
# 2.  MULTI-HEAD ATTENTION
# ─────────────────────────────────────────────────────────────────

class MultiHeadAttention(nn.Module):
    """
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W^O
    head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)

    Default: d_model=512, h=8, d_k=d_v=64
    """

    def __init__(self, d_model: int = 512, h: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.h = h
        self.d_k = d_model // h
        self.d_model = d_model

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.attn = ScaledDotProductAttention(dropout)
        self.attn_weights = None   # stored for inspection

    def _split_heads(self, x: torch.Tensor, B: int) -> torch.Tensor:
        """(B, seq, d_model) → (B, h, seq, d_k)"""
        return x.view(B, -1, self.h, self.d_k).transpose(1, 2)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B = Q.size(0)

        Q = self._split_heads(self.W_q(Q), B)   # (B, h, seq_q, d_k)
        K = self._split_heads(self.W_k(K), B)   # (B, h, seq_k, d_k)
        V = self._split_heads(self.W_v(V), B)   # (B, h, seq_k, d_k)

        x, self.attn_weights = self.attn(Q, K, V, mask)  # (B, h, seq_q, d_k)

        # Concat heads: (B, h, seq, d_k) → (B, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        return self.W_o(x)   # (B, seq_q, d_model)


# ─────────────────────────────────────────────────────────────────
# 3.  POSITION-WISE FEED-FORWARD NETWORK
# ─────────────────────────────────────────────────────────────────

class PositionwiseFFN(nn.Module):
    """
    FFN(x) = ReLU(x·W_1 + b_1)·W_2 + b_2
    Applied independently to each position.
    d_ff = 4 × d_model (2048 for base model).
    """

    def __init__(self, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ─────────────────────────────────────────────────────────────────
# 4.  ENCODER LAYER  (one of N identical layers)
# ─────────────────────────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """
    Sub-layer 1: Multi-Head Self-Attention + Add & LayerNorm
    Sub-layer 2: Position-wise FFN + Add & LayerNorm
    """

    def __init__(
        self,
        d_model: int = 512,
        h: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Self-attention with residual
        _attn = self.self_attn(x, x, x, src_mask)   # Q=K=V=x (self-attention)
        x = self.norm1(x + self.dropout(_attn))

        # FFN with residual
        _ffn = self.ffn(x)
        x = self.norm2(x + self.dropout(_ffn))

        return x   # (B, seq, d_model)


# ─────────────────────────────────────────────────────────────────
# 5.  FULL ENCODER STACK
# ─────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """
    Complete encoder: token embedding + sinusoidal PE + N encoder layers.

    Input:  token indices (B, seq_len)
    Output: contextual representations (B, seq_len, d_model)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute and cache sinusoidal PE
        self.register_buffer("pe", self._build_pe(max_len, d_model))

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        """Sinusoidal PE table: (1, max_len, d_model)"""
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)   # even dimensions
        pe[:, 1::2] = torch.cos(pos * div)   # odd dimensions
        return pe.unsqueeze(0)               # (1, max_len, d_model)

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Step 1: token embedding + scale
        x = self.embed(src) * math.sqrt(self.d_model)

        # Step 2: add positional encoding
        seq_len = src.size(1)
        x = x + self.pe[:, :seq_len, :]

        # Step 3: dropout on combined embedding
        x = self.dropout(x)

        # Step 4: N encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)

        # Step 5: final layer norm
        return self.norm(x)   # (B, seq_len, d_model)


# ─────────────────────────────────────────────────────────────────
# 6.  PADDING MASK UTILITY
# ─────────────────────────────────────────────────────────────────

def make_padding_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Returns (B, 1, 1, seq_len) mask.
    Value 1 = real token (keep), 0 = [PAD] (mask out).
    The extra dims broadcast correctly with (B, h, seq_q, seq_k) attention scores.
    """
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


# ─────────────────────────────────────────────────────────────────
# 7.  CLASSIFICATION HEAD EXAMPLE (e.g. BERT-style sentiment analysis)
# ─────────────────────────────────────────────────────────────────

class EncoderForClassification(nn.Module):
    """
    Encoder + linear classifier.
    Uses [CLS] token representation (position 0) for the classification decision.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, N, h, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor = None
    ) -> torch.Tensor:
        enc_out = self.encoder(src, src_mask)   # (B, seq, d_model)
        cls_repr = enc_out[:, 0, :]             # [CLS] token at position 0
        cls_repr = self.dropout(cls_repr)
        return self.classifier(cls_repr)         # (B, num_classes)


# ─────────────────────────────────────────────────────────────────
# 8.  QUICK SMOKE TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    VOCAB = 30522    # BERT vocabulary size
    BATCH = 4
    SEQ   = 32
    D     = 512

    # Build encoder
    encoder = Encoder(vocab_size=VOCAB, d_model=D, N=6, h=8, d_ff=2048)

    # Fake token IDs (0 = [PAD])
    src = torch.randint(1, VOCAB, (BATCH, SEQ))
    src[0, 28:] = 0   # pad last 4 tokens in first sequence

    # Build padding mask
    mask = make_padding_mask(src, pad_idx=0)

    # Forward pass
    enc_out = encoder(src, mask)
    print(f"Encoder output shape: {enc_out.shape}")   # (4, 32, 512)

    # Classification
    clf = EncoderForClassification(VOCAB, num_classes=2, d_model=D)
    logits = clf(src, mask)
    print(f"Logits shape: {logits.shape}")             # (4, 2)
