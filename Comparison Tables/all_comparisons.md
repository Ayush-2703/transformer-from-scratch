# Comparison Tables

## 1. RNN vs LSTM vs Transformer

| Feature | Vanilla RNN | LSTM | Transformer |
|---------|-------------|------|-------------|
| **Origin** | Elman (1990) | Hochreiter & Schmidhuber (1997) | Vaswani et al. (2017) |
| **Core mechanism** | `h_t = tanh(W_hh·h_{t-1} + W_xh·x_t)` | Forget/input/output gates on cell state | Self-attention: `softmax(QKᵀ/√d_k)·V` |
| **Memory** | Single hidden state h_t | Two states: h_t (short) + c_t (long) | No recurrent state; global attention matrix |
| **Gradient stability** | Unstable (vanishing/exploding) | Stable (constant error carousels) | Stable (LayerNorm + residuals; no BPTT) |
| **Parallelizable** | No — must process t before t+1 | No — same sequential constraint | Yes — all positions computed simultaneously |
| **Long-range dependency** | Fails beyond ~10–20 steps | Good (hundreds of steps) | Excellent (full sequence in one step) |
| **Complexity (time)** | O(n·d²) | O(n·d²) — higher constant | O(n²·d) attention + O(n·d²) FFN |
| **Complexity (memory)** | O(d) — single state | O(d) — two states | O(n²) — attention matrix |
| **Parameter count** | Low | ~4× RNN (3 gates + cell update) | Huge (base: ~65M, large: ~213M) |
| **Key weakness** | Vanishing gradient | Slow sequential training; hard to scale | Quadratic cost for long sequences |
| **Primary use today** | Legacy; simple time-series | Moderate-length sequences; constrained compute | LLMs, modern NLP, computer vision |

---

## 2. Encoder-Only vs Decoder-Only vs Encoder-Decoder

| Feature | Encoder-Only | Decoder-Only | Encoder-Decoder |
|---------|-------------|--------------|----------------|
| **Attention direction** | Bidirectional (full) | Causal / unidirectional | Enc: bidirectional; Dec: causal |
| **Cross-attention** | No | No | Yes |
| **Masked self-attention** | No | Yes | Yes (decoder only) |
| **Sub-layers per layer** | 2 | 2 | 2 (encoder) / 3 (decoder) |
| **Primary task** | Understanding / classification | Generation | Sequence-to-sequence |
| **Inference mode** | Single forward pass | Auto-regressive (token by token) | Encode once; decode auto-regressively |
| **Pre-training objective** | Masked Language Modeling (MLM) | Causal LM (CLM) | Span corruption (T5) / Denoising (BART) |
| **Context** | Full source visible at once | Left context only at each step | Full source; left target context |
| **Key models** | BERT, RoBERTa, DistilBERT | GPT-2/3/4, LLaMA, Claude, Gemini | T5, BART, MarianMT, mBART |
| **Best for** | NER, QA from passage, classification, similarity | Chat, code, open-ended generation, reasoning | Translation, summarization, structured generation |

---

## 3. Hyperparameters — Original Paper

| Hyperparameter | Symbol | Base Model | Large Model |
|----------------|--------|------------|-------------|
| Model dimension | d_model | 512 | 1024 |
| Feed-forward dim. | d_ff | 2048 | 4096 |
| Attention heads | h | 8 | 16 |
| Dim. per head | d_k = d_v | 64 | 64 |
| Encoder layers | N_enc | 6 | 6 |
| Decoder layers | N_dec | 6 | 6 |
| Dropout | p_drop | 0.1 | 0.3 |
| Label smoothing | ε | 0.1 | 0.1 |
| Warmup steps | — | 4,000 | 4,000 |
| β₁ (Adam) | — | 0.9 | 0.9 |
| β₂ (Adam) | — | 0.98 | 0.98 |
| ε (Adam) | — | 1e-9 | 1e-9 |
| Max gradient norm | — | — | — |
| Total parameters | — | ~65M | ~213M |
| Training steps | — | 100,000 | 300,000 |
| Training time (8×P100) | — | 12 hours | 3.5 days |

---

## 4. Modern Large Language Models

| Model | Architecture | Params | Context | Year |
|-------|-------------|--------|---------|------|
| BERT-base | Encoder-only | 110M | 512 | 2018 |
| GPT-2 (large) | Decoder-only | 774M | 1024 | 2019 |
| T5-large | Encoder-decoder | 770M | 512 | 2019 |
| GPT-3 | Decoder-only | 175B | 2048 | 2020 |
| BERT-large | Encoder-only | 340M | 512 | 2018 |
| LLaMA-2 (13B) | Decoder-only | 13B | 4096 | 2023 |
| Mistral-7B | Decoder-only | 7B | 32768 | 2023 |
| LLaMA-3 (70B) | Decoder-only | 70B | 8192 | 2024 |
| GPT-4 | Decoder-only | ~1T (est.) | 128K | 2023 |
| Gemini 1.5 Pro | Decoder-only | Unknown | 1M | 2024 |

---

## 5. Decoding Strategies Comparison

| Strategy | Deterministic? | Quality | Diversity | Best for |
|----------|---------------|---------|-----------|----------|
| Greedy | Yes | Moderate | Low | Simple tasks, fast inference |
| Beam search (B=4) | Yes | High | Low | Translation, summarization |
| Temperature sampling | No | Varies | High | Creative writing |
| Top-k (k=50) | No | Good | Moderate | General generation |
| Top-p (p=0.9) | No | Good | Adaptive | Production chatbots |
| Top-k + temperature | No | Good | Tunable | Most common in practice |

---

## 6. Positional Encoding Methods

| Method | Generalizes past max_len? | Relative position | Used by |
|--------|--------------------------|-------------------|---------|
| Sinusoidal (fixed) | Yes | Linear functions (learnable) | Original Transformer |
| Learned absolute | No | None | GPT-2, BERT |
| RoPE | Yes (with scaling tricks) | Native | LLaMA, Mistral, GPT-NeoX |
| ALiBi | Yes | Linear bias | MPT, BLOOM |
| NoPE | N/A | None (relies on ALiBi-like) | Some recent models |
