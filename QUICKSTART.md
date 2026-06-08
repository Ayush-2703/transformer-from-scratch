# Quick Start Guide

## Setup

```bash
git clone https://github.com/your-username/transformer-complete-guide
cd transformer-complete-guide
pip install -r requirements.txt
```

---

## Run the Encoder

```bash
cd Encoder_Transformer/Code
python encoder_full.py
```

Expected output:
```
Encoder output shape: torch.Size([4, 32, 512])
Logits shape: torch.Size([4, 2])
```

---

## Run the GPT Decoder (PyTorch)

```bash
cd Decoder_Transformer/Code
python gpt_decoder.py
```

Expected output:
```
Parameters: 44.2M
Logits shape: torch.Size([2, 32, 50257])
Loss: 10.8xxx
Generated sequence length: 25
```

---

## Run the Pure Python Decoder (No Libraries)

```bash
cd Decoder_Transformer/Code
python gpt_pure_python.py
```

This will:
1. Download `names.txt` (~32K names) if not present
2. Train a tiny Transformer decoder for 500 steps
3. Generate 20 name samples

Expected output (after training):
```
--- Generated samples ---
Sample  1: amelia
Sample  2: jordan
...
```

---

## Run the Full Encoder-Decoder

```bash
cd Encoder_Decoder_Transformer/Code
python enc_dec_transformer.py
```

Expected output:
```
Parameters: 12.0M
Logits: torch.Size([2, 15, 6000])
Label smoothed loss: 8.xxxx
Greedy translation shape: torch.Size([1, N])
```

---

## Run Attention Demos

```bash
cd Foundations/Attention_Mechanism
python attention_code.py
```

---

## Training with Utilities

```bash
cd Training/Code
python training_utils.py
```

Shows LR schedule at key steps:
```
Step    100: lr = 0.000218
Step    500: lr = 0.000490
Step   1000: lr = 0.000693
Step   2000: lr = 0.000693
Step   4000: lr = 0.000693
Step   8000: lr = 0.000490
Step  10000: lr = 0.000438
```

---

## Study Path

### Day 1 — Foundations
1. Read `Foundations/RNN/theory.md`
2. Read `Foundations/LSTM/theory.md`
3. Read `Foundations/Attention_Mechanism/theory.md`
4. Run `Foundations/Attention_Mechanism/attention_code.py`

### Day 2 — Encoder
1. Read `Foundations/Input_Embedding/theory_and_code.md`
2. Read `Foundations/Positional_Encoding/theory_and_code.md`
3. Read `Encoder_Transformer/Theory_and_Math/encoder_theory.md`
4. Read `Encoder_Transformer/Code_Explanation/encoder_explained.md`
5. Run `Encoder_Transformer/Code/encoder_full.py`

### Day 3 — Decoder
1. Read `Decoder_Transformer/Theory_and_Math/decoder_theory.md`
2. Read `Decoder_Transformer/Code_Explanation/decoder_explained.md`
3. Run `Decoder_Transformer/Code/gpt_decoder.py`
4. Run `Decoder_Transformer/Code/gpt_pure_python.py`

### Day 4 — Full Transformer
1. Read `Encoder_Decoder_Transformer/Theory_and_Math/enc_dec_theory.md`
2. Read `Encoder_Decoder_Transformer/Code_Explanation/enc_dec_explained.md`
3. Run `Encoder_Decoder_Transformer/Code/enc_dec_transformer.py`

### Day 5 — Training
1. Read `Training/Theory/training_theory.md`
2. Read `Training/Code_Explanation/training_explained.md`
3. Run `Training/Code/training_utils.py`

### Day 6 — Big Picture
1. Read `Comparison_Tables/all_comparisons.md`
2. Read `Applications/applications.md`
3. Read `Limitations/limitations.md`
4. Browse `Glossary/glossary.md` for any unclear terms
5. Check `Architecture_Diagrams/diagrams.md`

---

## File Count Summary

| Section | Files |
|---------|-------|
| Encoder_Transformer | 3 |
| Decoder_Transformer | 4 |
| Encoder_Decoder_Transformer | 3 |
| Foundations | 6 |
| Training | 3 |
| Comparison_Tables | 1 |
| Glossary | 1 |
| Limitations | 1 |
| Applications | 1 |
| Architecture_Diagrams | 1 |
| References | 1 |
| Root (README, requirements, quickstart) | 3 |
| **Total** | **~28 files** |
