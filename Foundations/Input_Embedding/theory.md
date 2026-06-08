# Input Embedding

## What Embeddings Do

Neural networks operate on continuous vectors, not discrete symbols. Input embedding converts each discrete token (an integer ID) into a dense vector of dimension d_model (512 in the original paper).

This is a learned lookup table: a matrix E of shape (vocab_size, d_model). Token ID `i` maps to the i-th row of E.

---

## Tokenization → Token ID → Embedding

```
Raw text:  "The cat sat"
           ↓  tokenizer (e.g. BPE)
Tokens:    ["The", "cat", "sat"]
           ↓  vocabulary lookup
IDs:       [1796, 3797, 3290]
           ↓  embedding matrix E (vocab_size × d_model)
Vectors:   [[0.12, -0.45, ...], [0.87, 0.21, ...], [-0.33, 0.94, ...]]
           shape: (3, 512)
```

---

## Tokenization Methods

| Method | Description | Used by |
|--------|-------------|---------|
| **BPE** (Byte Pair Encoding) | Merges frequent character pairs iteratively | GPT-2, RoBERTa |
| **WordPiece** | Similar to BPE, maximizes language model likelihood | BERT |
| **SentencePiece** | Language-agnostic subword tokenizer | T5, LLaMA |
| **Character-level** | One token per character | Some small models |

All methods produce a fixed vocabulary (e.g., 32K–50K tokens), so rare words become sequences of subword tokens rather than [UNK].

---

## What Embeddings Learn

Semantically related words cluster together in embedding space. The classic example:

```
king - man + woman ≈ queen
```

This arises naturally from training on large corpora — words appearing in similar contexts end up with similar vectors (the distributional hypothesis).

In the paper's example, "black" and "brown" have cosine similarity ≈ 0.9999 in embedding space after training.

---

## Embedding Scale

The original paper multiplies embeddings by √d_model before adding positional encoding:

```
x = embed(token_id) * sqrt(d_model)
x = x + positional_encoding
```

**Why?** Embedding weights are initialized with values near 0 (std ≈ 1/√d_model). Positional encodings are bounded in [-1, 1]. Without scaling, the PE would dominate. Multiplying by √d_model brings embeddings and PE to a comparable magnitude.

---

## Weight Tying

In many modern models (GPT-2, T5), the input embedding matrix E and the output projection matrix W_out share the same weights:

```
logits = decoder_output · E^T    (reuse the same E transposed)
```

**Benefits:**
- Reduces parameter count by vocab_size × d_model (25M+ for large vocabularies)
- Often improves performance: the output projection must be consistent with the input embedding space
- Provides a natural regularizer

---
