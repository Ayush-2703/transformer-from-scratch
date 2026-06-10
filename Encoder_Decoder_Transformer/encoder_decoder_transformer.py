"""
Full Encoder-Decoder Transformer — PyTorch Implementation.
Based on Vaswani et al. (2017) "Attention Is All You Need".
Covers: FullDecoderLayer (3 sub-layers with cross-attention),
        EncoderDecoderTransformer, LabelSmoothingLoss,
        training step, greedy decoding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ─────────────────────────────────────────────────────────────────
# Import shared components (copy from encoder_full.py or import)
# ─────────────────────────────────────────────────────────────────

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(weights, V), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, h=8, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.h, self.d_k, self.d_model = h, d_model // h, d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attn = ScaledDotProductAttention(dropout)

    def _split(self, x, B):
        return x.view(B, -1, self.h, self.d_k).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        Q = self._split(self.W_q(Q), B)
        K = self._split(self.W_k(K), B)
        V = self._split(self.W_v(V), B)
        x, _ = self.attn(Q, K, V, mask)
        return self.W_o(x.transpose(1, 2).contiguous().view(B, -1, self.d_model))


class PositionwiseFFN(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, N=6, h=8, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, h, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pe", self._build_pe(max_len, d_model))

    @staticmethod
    def _build_pe(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, src, src_mask=None):
        x = self.embed(src) * math.sqrt(self.d_model)
        x = x + self.pe[:, :src.size(1), :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────
# FULL DECODER LAYER  (3 sub-layers including cross-attention)
# ─────────────────────────────────────────────────────────────────

class FullDecoderLayer(nn.Module):
    """
    Sub-layer 1: Masked multi-head self-attention (causal)
    Sub-layer 2: Cross-attention (Q=decoder, K=V=encoder memory)
    Sub-layer 3: FFN
    Each sub-layer: residual + LayerNorm.
    """

    def __init__(self, d_model=512, h=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.ffn        = PositionwiseFFN(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _causal_mask(T, device):
        return torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

    def forward(self, tgt, memory, src_mask=None):
        """
        tgt:      (B, T_tgt, d_model)  — decoder input
        memory:   (B, T_src, d_model)  — encoder output (fixed)
        src_mask: (B, 1, 1, T_src)    — source padding mask
        """
        T, device = tgt.size(1), tgt.device
        tgt_mask = self._causal_mask(T, device)

        # Step 1: Masked self-attention on target sequence
        tgt = self.norm1(tgt + self.dropout(
            self.self_attn(tgt, tgt, tgt, tgt_mask)
        ))

        # Step 2: Cross-attention — Q from decoder, K,V from encoder
        tgt = self.norm2(tgt + self.dropout(
            self.cross_attn(tgt, memory, memory, src_mask)  # no causal mask here!
        ))

        # Step 3: FFN
        tgt = self.norm3(tgt + self.dropout(self.ffn(tgt)))

        return tgt


# ─────────────────────────────────────────────────────────────────
# FULL ENCODER-DECODER TRANSFORMER
# ─────────────────────────────────────────────────────────────────

class EncoderDecoderTransformer(nn.Module):
    """
    Full Transformer (Vaswani et al., 2017).
    Input:  src (B, src_len), tgt (B, tgt_len)
    Output: logits (B, tgt_len, tgt_vocab_size)
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        N: int = 6,
        h: int = 8,
        d_ff: int = 2048,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Encoder
        self.encoder = Encoder(src_vocab_size, d_model, N, h, d_ff, max_len, dropout)

        # Decoder components
        self.tgt_embed   = nn.Embedding(tgt_vocab_size, d_model)
        self.dec_layers  = nn.ModuleList(
            [FullDecoderLayer(d_model, h, d_ff, dropout) for _ in range(N)]
        )
        self.dec_norm    = nn.LayerNorm(d_model)
        self.dec_dropout = nn.Dropout(dropout)

        # Output projection
        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

        # Sinusoidal PE (shared by encoder and decoder)
        self.register_buffer("pe", Encoder._build_pe(max_len, d_model))

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask=None) -> torch.Tensor:
        """Returns encoder memory: (B, src_len, d_model)"""
        return self.encoder(src, src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask=None) -> torch.Tensor:
        """
        Runs decoder layers.
        tgt:    (B, tgt_len) — target token IDs
        memory: (B, src_len, d_model) — encoder output
        """
        tgt_len = tgt.size(1)
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = x + self.pe[:, :tgt_len, :]
        x = self.dec_dropout(x)

        for layer in self.dec_layers:
            x = layer(x, memory, src_mask)

        return self.dec_norm(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask=None) -> torch.Tensor:
        """
        Full forward pass.
        Training: tgt_in = tgt[:, :-1], tgt_out = tgt[:, 1:]
        """
        memory  = self.encode(src, src_mask)
        dec_out = self.decode(tgt, memory, src_mask)
        return self.output_proj(dec_out)   # (B, tgt_len, tgt_vocab_size)


# ─────────────────────────────────────────────────────────────────
# LABEL SMOOTHING LOSS
# ─────────────────────────────────────────────────────────────────

class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with label smoothing (ε=0.1 in original paper).
    True class gets prob (1-ε); remaining ε distributed uniformly.
    Prevents model from being overconfident → better generalization.
    """

    def __init__(self, vocab_size: int, pad_idx: int = 0, epsilon: float = 0.1):
        super().__init__()
        self.epsilon  = epsilon
        self.pad_idx  = pad_idx
        self.vocab_size = vocab_size

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)   # (N, V)

        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.epsilon / (self.vocab_size - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon)
            smooth[targets == self.pad_idx] = 0.0   # ignore padding

        loss = -(smooth * log_probs).sum() / (targets != self.pad_idx).sum().float()
        return loss


# ─────────────────────────────────────────────────────────────────
# TRAINING STEP
# ─────────────────────────────────────────────────────────────────

def train_step(model, optimizer, src, tgt, criterion, pad_idx=0):
    """
    One training step with teacher forcing.
    tgt: (B, tgt_len+1) — includes BOS and EOS
    """
    model.train()

    tgt_in  = tgt[:, :-1]   # BOS + words → decoder input
    tgt_out = tgt[:, 1:]    # words + EOS → labels

    # Source padding mask: (B, 1, 1, src_len)
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    logits = model(src, tgt_in, src_mask)   # (B, T, vocab)

    loss = criterion(
        logits.view(-1, logits.size(-1)),
        tgt_out.reshape(-1),
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


# ─────────────────────────────────────────────────────────────────
# GREEDY DECODING FOR INFERENCE
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def translate_greedy(model, src, bos_idx, eos_idx, max_len=100, pad_idx=0):
    """
    Greedy decoding: always pick the highest-probability next token.
    """
    model.eval()
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    memory   = model.encode(src, src_mask)   # encode source once

    B = src.size(0)
    tgt = torch.full((B, 1), bos_idx, dtype=torch.long, device=src.device)

    for _ in range(max_len):
        dec_out = model.decode(tgt, memory, src_mask)
        logits  = model.output_proj(dec_out[:, -1, :])   # (B, vocab) — last position
        next_token = logits.argmax(dim=-1, keepdim=True)   # (B, 1)
        tgt = torch.cat([tgt, next_token], dim=1)

        if (next_token.squeeze(-1) == eos_idx).all():
            break

    return tgt   # (B, generated_length)


# ─────────────────────────────────────────────────────────────────
# BEAM SEARCH DECODING (single-sequence, for clarity)
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def beam_search(model, src, bos_idx, eos_idx, beam_size=4, max_len=100,
                length_penalty=0.7, pad_idx=0):
    """
    Beam search for one source sequence (B=1).
    Returns list of (score, token_ids) sorted by score.
    """
    model.eval()
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    memory   = model.encode(src, src_mask)

    # Initial beam: one sequence starting with BOS
    beams = [(0.0, [bos_idx])]
    finished = []

    for _ in range(max_len):
        all_candidates = []

        for score, tokens in beams:
            if tokens[-1] == eos_idx:
                finished.append((score, tokens))
                continue

            tgt = torch.tensor([tokens], device=src.device)
            dec_out = model.decode(tgt, memory, src_mask)
            logits  = model.output_proj(dec_out[:, -1, :])        # (1, vocab)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)   # (vocab,)

            topk_log_probs, topk_ids = log_probs.topk(beam_size)
            for lp, tid in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                all_candidates.append((score + lp, tokens + [tid]))

        if not all_candidates:
            break

        # Keep top beam_size candidates (with length penalty)
        all_candidates.sort(
            key=lambda x: x[0] / (len(x[1]) ** length_penalty),
            reverse=True
        )
        beams = all_candidates[:beam_size]

    finished.extend(beams)
    finished.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
    return finished


# ─────────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SRC_VOCAB = 8000
    TGT_VOCAB = 6000

    model = EncoderDecoderTransformer(SRC_VOCAB, TGT_VOCAB, d_model=256, N=3, h=8, d_ff=512)
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.1f}M")

    # Simulate a translation batch: src in English, tgt in French
    src = torch.randint(1, SRC_VOCAB, (2, 20))   # batch=2, src_len=20
    tgt = torch.randint(1, TGT_VOCAB, (2, 16))   # tgt_len=16 (includes BOS/EOS)

    src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

    logits = model(src, tgt[:, :-1], src_mask)
    print(f"Logits: {logits.shape}")   # (2, 15, 6000)

    # Label smoothing loss
    criterion = LabelSmoothingLoss(TGT_VOCAB, pad_idx=0, epsilon=0.1)
    loss = criterion(logits.view(-1, TGT_VOCAB), tgt[:, 1:].reshape(-1))
    print(f"Label smoothed loss: {loss.item():.4f}")

    # Greedy decoding
    translated = translate_greedy(model, src[:1], bos_idx=1, eos_idx=2, max_len=20)
    print(f"Greedy translation shape: {translated.shape}")
