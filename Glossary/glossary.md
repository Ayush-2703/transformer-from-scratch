# Glossary — All Transformer Terms

## A

**Attention (general):** A mechanism that computes a weighted sum of a set of value vectors, where the weights depend on the compatibility between a query vector and a set of key vectors.

**Attention head:** One independent attention computation in a multi-head attention layer. The d_model-dimensional input is split across h heads, each operating in a d_k-dimensional subspace.

**Auto-regressive:** A property of a generative model where each output token is conditioned on all previously generated tokens. Essential for language generation.

**ALiBi (Attention with Linear Biases):** A positional encoding method that adds a position-dependent bias directly to attention scores, enabling generalization to longer sequences than seen during training.

---

## B

**Backpropagation Through Time (BPTT):** The algorithm for training RNNs by unrolling the network across all time steps and applying standard backpropagation. Causes vanishing/exploding gradient problems for long sequences.

**Beam search:** A decoding strategy that maintains B candidate sequences ("beams") at each step, extending each beam with all vocabulary tokens and keeping the top B by cumulative log-probability.

**BPE (Byte Pair Encoding):** A subword tokenization algorithm that iteratively merges the most frequent pair of characters or character sequences. Used by GPT-2, RoBERTa.

**BERT:** Bidirectional Encoder Representations from Transformers. An encoder-only Transformer pre-trained with Masked Language Modeling (MLM). State-of-the-art for understanding tasks.

---

## C

**Causal mask:** A lower-triangular binary mask applied in the decoder's self-attention that prevents position i from attending to positions j > i. Enforces the auto-regressive property during parallel training.

**Cell state (c_t):** The long-term memory component in an LSTM. Modified only via gated addition, providing a direct gradient path through time.

**Context window:** The maximum number of tokens a Transformer can process at once. Beyond this limit, the model cannot attend to earlier tokens.

**Cross-attention:** An attention mechanism in the encoder-decoder architecture where Queries come from the decoder and Keys/Values come from the encoder output. Allows the decoder to consult the source sequence at every generation step.

---

## D

**d_ff:** The inner dimension of the feed-forward network. Typically 4× d_model (2048 for the base model).

**d_k:** The dimension of Query and Key vectors in each attention head. d_k = d_model / h.

**d_model:** The main embedding dimension of the Transformer (512 for the base model). All sub-layers maintain this dimension.

**d_v:** The dimension of Value vectors in each attention head. Usually d_v = d_k = d_model / h.

**Decoder:** The Transformer component that generates output sequences auto-regressively. Can be standalone (GPT) or combined with an encoder (T5).

**Dropout:** A regularization technique that randomly zeros a fraction p of activations during training. Applied to embeddings, sub-layer outputs, and attention weights.

---

## E

**Embedding:** A dense, continuous vector representation of a discrete token. Learned during training; tokens with similar contexts end up with similar embeddings.

**Encoder:** The Transformer component that reads the full input sequence and produces contextual representations for every token position. No causal mask — fully bidirectional.

**Encoder memory:** The output of the encoder stack. A matrix of shape (batch, src_len, d_model). Passed to every decoder layer via cross-attention.

**Exposure bias:** The mismatch between training (teacher forcing — sees correct previous tokens) and inference (sees its own previous predictions). Can cause error compounding.

---

## F

**Feed-forward network (FFN):** A two-layer MLP applied independently to each token position: `FFN(x) = max(0, xW₁ + b₁)W₂ + b₂`. Provides non-linear transformation within each layer.

**Forget gate (f_t):** LSTM gate that decides what fraction of the previous cell state to keep: `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)`.

**Foundation model:** A large model pre-trained on vast data that can be fine-tuned or prompted for many downstream tasks without task-specific architecture changes.

---

## G

**GPT (Generative Pre-trained Transformer):** A family of decoder-only Transformers pre-trained with Causal Language Modeling (CLM). Used for text generation, code, dialogue.

**Gradient clipping:** Rescaling all gradients if the global gradient norm exceeds a threshold (typically 1.0). Prevents catastrophically large updates.

**GRU (Gated Recurrent Unit):** A simplified LSTM with two gates (update and reset) and no separate cell state. Comparable performance with fewer parameters.

---

## H

**h:** Number of attention heads. Original paper: h=8.

**Hallucination:** The tendency of LLMs to generate fluent, confident-sounding but factually incorrect content. A fundamental limitation of pattern-matching without world models.

**Hidden state (h_t):** The recurrent state in an RNN or LSTM. Carries information from previous time steps. In LSTMs, also called the "short-term memory."

---

## I

**Input gate (i_t):** LSTM gate controlling how much new information is written to the cell state: `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)`.

---

## K

**Key (K):** The "what I have to offer" component in attention. Q·Kᵀ computes compatibility scores between queries and keys.

**KV cache:** During auto-regressive inference, previously computed Key and Value vectors are cached and reused rather than recomputed at each step. Critical for efficient generation.

---

## L

**Label smoothing:** A regularization technique that replaces one-hot targets with a soft distribution: true class gets 1-ε, others get ε/(V-1). Original paper: ε=0.1.

**Layer normalization (LayerNorm):** Normalizes the d_model features at each position independently. Applied after each sub-layer (Post-LN) or before (Pre-LN, more stable for deep models).

**LLM (Large Language Model):** A decoder-only Transformer trained at scale (billions of parameters, trillions of tokens). Exhibits emergent capabilities not present in smaller models.

**LSTM (Long Short-Term Memory):** An RNN variant with three gates (forget, input, output) and a separate cell state, designed to handle long-range dependencies.

---

## M

**Masked Language Modeling (MLM):** BERT's pre-training objective. Random tokens in the input are masked, and the model must predict the original tokens using bidirectional context.

**Multi-head attention:** Running h independent attention heads in parallel, each in a d_k-dimensional subspace, then concatenating and projecting the results.

---

## N

**Nucleus sampling (top-p):** Sample from the smallest set of tokens whose cumulative probability exceeds p. Adapts dynamically to the model's confidence.

---

## O

**Output gate (o_t):** LSTM gate controlling how much of the cell state is exposed as the hidden state: `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)`.

---

## P

**Padding mask:** A mask that prevents the model from attending to [PAD] tokens added to shorter sequences in a batch.

**Perplexity:** `exp(cross-entropy loss)`. A measure of how well a language model predicts a sequence. Lower is better. A perplexity of N means the model is as uncertain as choosing uniformly from N options.

**Positional encoding (PE):** Information added to token embeddings to inject sequence order. The Transformer is permutation-invariant without it.

**Pre-training:** Training a model on large amounts of unlabeled data with a self-supervised objective (MLM, CLM). The resulting model can be fine-tuned for specific tasks.

**Prompt engineering:** Designing input text to elicit desired behavior from a pre-trained model, without updating model weights.

---

## Q

**Query (Q):** The "what I'm looking for" component in attention. Compatibility with Keys determines attention weights.

---

## R

**Residual connection:** Adding the input of a sub-layer directly to its output: `x = x + Sublayer(x)`. Provides a gradient highway through deep networks.

**RMSNorm:** Root Mean Square normalization. Simpler than LayerNorm (no mean subtraction, no learned bias). Used by LLaMA, Mistral.

**RNN (Recurrent Neural Network):** A neural network that processes sequences by maintaining a hidden state updated at each time step.

**RoPE (Rotary Position Embedding):** Encodes position by rotating Query and Key vectors. The dot product depends on relative position only. Used by LLaMA, Mistral, GPT-NeoX.

---

## S

**Scaled dot-product attention:** `Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V`. The fundamental attention operation. Scaling by √d_k prevents softmax saturation for large d_k.

**Self-attention:** Attention where Q, K, and V all come from the same sequence. Each token attends to every other token in the same sequence.

**Seq2seq:** Sequence-to-sequence tasks: the input is one sequence, the output is a different sequence. Translation, summarization, etc.

---

## T

**Teacher forcing:** During training, the decoder receives correct previous tokens rather than its own predictions. Enables parallel training but causes exposure bias.

**Temperature:** A parameter T dividing logits before softmax. T<1 → sharper (more deterministic). T>1 → flatter (more diverse).

**Tokenization:** Splitting raw text into discrete units (tokens). Common methods: BPE, WordPiece, SentencePiece.

**Top-k sampling:** Sample from the k highest-probability tokens only. Prevents very low-probability tokens from being selected.

---

## V

**Value (V):** The "actual content" component in attention. The output is a weighted sum of Value vectors, where weights come from Q-K compatibility.

**Vanishing gradient:** A training problem where gradients become exponentially small as they propagate backward through many layers or time steps, preventing earlier parameters from learning.

**Vision Transformer (ViT):** Applies the Transformer encoder to sequences of image patches. Achieves state-of-the-art image classification on large datasets.

---

## W

**Warmup:** A learning rate schedule phase where lr increases from near-zero to its peak over a fixed number of steps. Prevents large early updates when gradients are unreliable.

**Weight tying:** Sharing the token embedding matrix and the output projection matrix. Reduces parameters and often improves performance.

**WordPiece:** A subword tokenization algorithm used by BERT that maximizes the likelihood of the training data.
