# Attention Mechanism

## The Core Idea

Before attention, encoder-decoder RNNs compressed the entire source sequence into a single fixed-size vector. This vector had to carry all meaning — which works for short sentences but breaks down for longer ones.

Attention says: at each decoding step, *look back at all encoder states* and take a weighted average. The weights are computed dynamically based on how relevant each source position is to the current decoding step.

---

## Query, Key, Value Framework

The Q/K/V abstraction generalises attention into a retrieval system:

| Component | Meaning | Origin |
|-----------|---------|--------|
| **Query (Q)** | "What am I looking for?" | Decoder state (or current token) |
| **Key (K)** | "What do I have to offer?" | Each encoder state (or all tokens) |
| **Value (V)** | "What is the actual content?" | Each encoder state (or all tokens) |

Step 1: Compute a score for every (Q, K) pair  
Step 2: Normalize scores with softmax → attention weights  
Step 3: Weighted sum of V vectors → context vector

---

## Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax( QKᵀ / √d_k ) · V
```

**Why the √d_k scaling?**  
When d_k is large, dot products grow in magnitude. Large values push softmax into regions with near-zero gradients. Dividing by √d_k keeps the pre-softmax values in a stable range.

### Shape flow:
```
Q: (batch, seq_q, d_k)
K: (batch, seq_k, d_k)
V: (batch, seq_k, d_v)

QKᵀ:         (batch, seq_q, seq_k)   ← score for every Q-K pair
softmax(·):  (batch, seq_q, seq_k)   ← attention weights
output:      (batch, seq_q, d_v)     ← weighted sum of V
```

---

## Multi-Head Attention

Single-head attention uses one projection of Q, K, V. Multi-head runs h independent attention heads in parallel, each looking at the input from a different learned subspace, then concatenates results:

```
head_i = Attention(Q · W_i^Q,  K · W_i^K,  V · W_i^V)

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O
```

**Original paper parameters:**
- h = 8 heads
- d_model = 512
- d_k = d_v = d_model / h = 64 per head

The total computation is similar to single-head attention with full d_model, but each head specialises in different relationship types (syntax, coreference, proximity, etc.).

---

## Three Uses of Attention in the Transformer

| Type | Q source | K, V source | Mask |
|------|----------|-------------|------|
| Encoder self-attention | Encoder layer output | Same | None (bidirectional) |
| Decoder masked self-attention | Decoder layer output | Same | Causal (lower-triangular) |
| Cross-attention | Decoder | Encoder memory | None (decoder sees all encoder positions) |

---

## Bahdanau Attention (Historical Context)

The first attention mechanism, used inside RNN seq2seq models (2015):

```
e_{t,i} = v^T · tanh(W_s · s_{t-1} + W_h · h_i)
α_{t,i} = softmax(e_{t,i})
context_t = Σ_i α_{t,i} · h_i
```

Where s_{t-1} is the decoder hidden state and h_i are encoder hidden states. This is additive (not dot-product) attention. The Transformer replaced this with scaled dot-product attention, which is more parallelizable.

---

## Causal (Look-Ahead) Mask

Used in the decoder to prevent position i from attending to positions j > i:

```
M[i, j] = { -∞   if j > i
           {  0    otherwise

Masked attention = softmax((QKᵀ / √d_k) + M) · V
```

After adding -∞ to future positions, softmax assigns them exactly 0 weight. The model cannot cheat by seeing future tokens during training.

---

## Padding Mask

When batching sequences of different lengths, shorter sequences are padded with a [PAD] token. The model should not attend to padding:

```
mask[b, i] = { -∞   if token i is [PAD]
             {  0    otherwise
```

Applied by adding the mask to the raw attention scores before softmax.

---

## Attention Complexity

| Operation | Time | Memory |
|-----------|------|--------|
| QKᵀ computation | O(n² · d_k) | O(n²) |
| Softmax | O(n²) | O(n²) |
| Attention · V | O(n² · d_v) | O(n²) |
| **Total** | **O(n² · d)** | **O(n²)** |

This quadratic cost in sequence length n is the main limitation for long documents.
