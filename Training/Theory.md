# Training Transformers — Theory

## Overview

A correctly implemented Transformer architecture still fails without the right training setup. This section covers every component of the training pipeline used in the original paper and modern practice.

---

## 1. The Adam Optimizer

Transformers use **Adam** (Adaptive Moment Estimation) — a variant of SGD that maintains per-parameter adaptive learning rates.

### How Adam Works

Adam tracks two running averages per parameter:
- **m_t** (first moment): exponential moving average of gradients (momentum)
- **v_t** (second moment): exponential moving average of squared gradients (adaptive LR)

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t          # momentum
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²         # squared gradient

m̂_t = m_t / (1 - β₁ᵗ)                         # bias correction
v̂_t = v_t / (1 - β₂ᵗ)                         # bias correction

θ_t = θ_{t-1} - lr_t · m̂_t / (√v̂_t + ε)
```

### Hyperparameters (Original Transformer)

| Parameter | Value | Role |
|-----------|-------|------|
| β₁ | 0.9 | Momentum decay (gradient direction) |
| β₂ | 0.98 | Second-moment decay (conservative; stabilizes warmup) |
| ε | 10⁻⁹ | Numerical stability (prevents divide-by-zero) |

Note: β₂=0.98 is more conservative than the default 0.999. The second moment needs to stabilize quickly during the warmup phase, where gradients change dramatically.

### Why Not Vanilla SGD?

Vanilla SGD uses one global learning rate for all parameters. In Transformers:
- Embedding parameters and attention weights operate at very different scales
- Some parameters are updated rarely (infrequent tokens in embedding matrix)
- Adam's per-parameter adaptive rates handle this naturally

---

## 2. Learning Rate Schedule with Warmup

The single most important training trick in the original Transformer paper.

```
lr(step) = d_model^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))
```

### Two Phases

**Phase 1 — Warmup (steps 1 to warmup_steps):**
```
lr ≈ step / warmup_steps^(1.5) · d_model^(-0.5)     (increases linearly)
```

**Phase 2 — Decay (steps > warmup_steps):**
```
lr ≈ step^(-0.5) · d_model^(-0.5)                    (decreases as 1/√step)
```

### Why Warmup?

Early in training, parameters are random and gradient directions are unreliable. Starting with a large learning rate → large noisy updates → model lands in a bad region of the loss landscape.

Warmup starts with lr ≈ 0, gradually increasing as the optimizer accumulates reliable gradient statistics. By the time lr is large, gradients are pointing in useful directions.

### Peak LR for Base Model

At step = warmup_steps = 4000:
```
lr_peak = 4000^(-0.5) · 512^(-0.5) ≈ 0.000698 ≈ 7×10⁻⁴
```

---

## 3. Dropout as Regularization

Applied at three points throughout the Transformer:

| Location | Purpose |
|----------|---------|
| Embedding + PE sum | Prevents over-reliance on specific positions |
| Each sub-layer output (before residual add) | Prevents co-adaptation of features |
| Attention weights | Prevents attending to same positions every forward pass |

Rate: p=0.1 for base model, p=0.3 for large model.

During training, each dropout mask randomly zeros 10% of values. During inference, dropout is disabled and all values are used (implicitly averaging over all possible masks).

---

## 4. Label Smoothing

Standard training uses one-hot targets: true class gets probability 1, all others get 0. The model is penalized for any probability mass on "wrong" classes, even tiny amounts.

**Problem:** this encourages extreme overconfidence. The model learns to make logits very large for the true class, which can hurt generalization.

**Label smoothing** (ε=0.1): instead of hard 0/1 targets:
```
y_smooth[true_class]  = 1 - ε = 0.9
y_smooth[other class] = ε / (V - 1) ≈ 0.0001  (for V=30000)
```

Effect:
- Increases training loss (harder to satisfy)
- Decreases model calibration error
- Improves BLEU scores on translation
- Acts as regularization: prevents logits from growing without bound

---

## 5. Gradient Clipping

If gradient norms spike (especially early in training), parameter updates can be catastrophically large. Gradient clipping caps the global gradient norm:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

How it works:
1. Compute global norm: `||g|| = sqrt(Σ_i ||g_i||²)`
2. If `||g|| > max_norm`: scale all gradients by `max_norm / ||g||`
3. Otherwise: leave gradients unchanged

This is a safety mechanism. In a well-tuned training run, clipping rarely fires. If clipping fires frequently, it indicates an LR schedule or architecture problem.

---

## 6. Hardware and Batching

**Original paper:** 8 × NVIDIA P100 GPUs
- Base model: 100K steps, ~12 hours
- Large model: 300K steps, ~3.5 days
- Batch size: ~25K source tokens + ~25K target tokens per update

**Modern practice:**
- Much larger batches (hundreds of thousands of tokens)
- Data parallelism: same model on multiple GPUs, gradients averaged
- Gradient accumulation: simulate large batches on limited hardware by accumulating gradients over N steps before updating
- Mixed precision: FP16/BF16 for activations, FP32 for optimizer state

### Gradient Accumulation

```python
accumulation_steps = 8   # simulate batch 8x larger

optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
```

---

## 7. Pre-Layer Normalization vs Post-Layer Normalization

**Original paper (Post-LN):**
```
x = LayerNorm(x + Sublayer(x))
```

**Modern practice (Pre-LN):**
```
x = x + Sublayer(LayerNorm(x))
```

Pre-LN is more stable for very deep models and does not require warmup as carefully. Most modern LLMs (GPT-3, LLaMA, etc.) use Pre-LN. Post-LN can achieve slightly better final performance but is harder to train.

---

## 8. Full Training Recipe Summary

```
Optimizer:          Adam (β₁=0.9, β₂=0.98, ε=1e-9)
LR schedule:        warmup (4000 steps) then inverse sqrt decay
Dropout:            0.1 (base), 0.3 (large)
Label smoothing:    ε=0.1
Gradient clipping:  max_norm=1.0
Batch size:         ~25K tokens per source/target
Weight init:        Xavier uniform for linear layers
Layer norm:         Post-LN (original) or Pre-LN (modern)
```
