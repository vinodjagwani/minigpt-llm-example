# MiniGPT — Character-Level Language Model in PyTorch

A minimal but complete transformer-based LLM built from scratch. Trains on Shakespeare text and generates new text in the same style.

---

## What You'll Learn

- How a **Transformer** works (attention, residual connections, layer norm)
- How **causal (masked) self-attention** prevents the model from "seeing the future"
- How a **character-level tokenizer** works
- The full **training loop**: forward pass → loss → backward → optimizer step
- How to **sample/generate** text from a trained model

---

## Architecture Overview

```
Input tokens
     │
     ▼
Token Embedding + Positional Embedding
     │
     ▼
┌─────────────────────────┐
│   Transformer Block ×3  │
│  ┌───────────────────┐  │
│  │  LayerNorm        │  │
│  │  CausalAttention  │  │  ← masked: each token only sees past tokens
│  │  + Residual       │  │
│  ├───────────────────┤  │
│  │  LayerNorm        │  │
│  │  Feed-Forward     │  │  ← two linear layers + GELU
│  │  + Residual       │  │
│  └───────────────────┘  │
└─────────────────────────┘
     │
     ▼
LayerNorm → Linear → logits (vocab_size)
     │
     ▼
Cross-Entropy Loss (during training)
     or
Softmax → Sample next token (during generation)
```

---

## Key Concepts Explained

### 1. Tokenization (Character-Level)
Each unique character gets an integer ID. "hello" → [7, 4, 11, 11, 14].
Real LLMs use subword tokenizers (BPE) — same idea, larger vocab.

### 2. Embeddings
- **Token embedding**: maps each integer ID to a vector (learned)
- **Positional embedding**: adds position info so the model knows token order

### 3. Causal Self-Attention
The core of the Transformer. Each token asks "which past tokens are relevant to me?"
- Q (query), K (key), V (value) — all derived from the same input
- Score = Q @ Kᵀ / √head_dim — how much each token attends to others
- **Causal mask**: upper-triangle set to -∞ so softmax zeroes out future positions
- Output = weighted sum of V vectors

### 4. Multi-Head Attention
Run attention H times in parallel with different learned projections.
Each "head" can focus on different relationships (syntax, semantics, etc.).

### 5. Feed-Forward Network
Two linear layers with GELU activation. Applied identically at each position.
Increases model capacity — most parameters live here.

### 6. Residual Connections + LayerNorm
`x = x + sublayer(LayerNorm(x))` — stabilizes training, enables deep networks.

### 7. Weight Tying
Input embedding and output projection share weights — reduces parameters and often improves quality.

### 8. Generation
```
prompt → model → logits → temperature scaling → top-k filtering → softmax → sample
                                                                            │
                                         append to sequence ←──────────────┘
                                         repeat until done
```
- **Temperature**: higher = more random, lower = more deterministic
- **Top-k**: only sample from the k most likely next tokens

---

## Files

```
minigpt-llm-example/
├── model.py     — MiniGPT architecture (attention, transformer block, generation)
├── train.py     — data loading, training loop, text generation demo
├── generate.py  — load trained model and generate text interactively
└── README.md    — this file
```

---

## Requirements

```bash
pip install torch requests
```

Python 3.8+ required. GPU optional but faster.

---

## Run

```bash
cd minigpt-llm-example
python train.py
```

Expected output:
```
Device: cpu
Downloading Shakespeare dataset...
Corpus: 1,115,394 chars | vocab: 65 unique chars
Model params: 1,678,209
step    1/2000  train=4.1732  val=4.1745
step  200/2000  train=2.1803  val=2.2541
step  400/2000  train=1.9201  val=2.0988
...
step 2000/2000  train=1.6843  val=1.8502

--- Generated text ---
GLOUCESTER:
Come, let us go; I will not stay behind.
...
```

Loss ~1.65 means the model has learned structure. Shakespeare-quality requires much more compute.

---

## Generate Text (after training)

```bash
# basic — generates from blank prompt
python generate.py

# custom prompt
python generate.py --prompt "My lord"
python generate.py --prompt "HAMLET:"
python generate.py --prompt "I am the king"
```

### Prompt Examples

| Prompt | Style |
|--------|-------|
| `"My lord"` | Dialogue opener |
| `"HAMLET:"` | Character speech |
| `"BRUTUS:"` | Character speech |
| `"OTHELLO:"` | Character speech |
| `"QUEEN:"` | Royalty dialogue |
| `"RICHARD:"` | Character speech |
| `"The king shall"` | Narrative |
| `"I love thee"` | Romantic |
| `"Alas, poor"` | Famous line continuation |
| `"All the world"` | Famous line continuation |
| `"Friends, Romans"` | Famous line continuation |
| `"Draw thy sword"` | Conflict scene |
| `"Traitor! Thou"` | Conflict scene |
| `"To war, to"` | Battle scene |

### Sample Output

```
I am the kings. I am one of him
Had he princes to make understand.

CLIFFORD:
Well, give me the fatal villain?

DUCHESS OF YORK:
My Lord of Gloucester, with the judgments
But hence and from the ground tender root.

BRUTUS:
Ay, nay, tell me, the king's exclime,
Where is your truth, it is not to be your tender,
```

### Generation Flags

```bash
python generate.py --prompt "HAMLET:" --tokens 500   # longer output
python generate.py --prompt "My lord" --temp 0.3     # focused, repetitive
python generate.py --prompt "My lord" --temp 0.8     # balanced (default)
python generate.py --prompt "My lord" --temp 1.2     # creative, wilder
python generate.py --prompt "My lord" --top_k 5      # very focused sampling
python generate.py --prompt "My lord" --top_k 40     # diverse sampling (default)
```

### Temperature Guide

| Temperature | Effect |
|-------------|--------|
| `0.1–0.4` | Repetitive but coherent |
| `0.7–0.9` | Best balance (recommended) |
| `1.0–1.5` | Creative but more chaotic |

> **Note:** Use Shakespeare-style prompts only. Modern words (`hello`, `hey`, `okay`) produce worse output — model only knows what it saw in training data.

---

## Hyperparameter Guide

| Parameter    | Default | Effect |
|--------------|---------|--------|
| `BLOCK_SIZE` | 128     | Context window. Bigger = more memory, richer context |
| `EMBED_DIM`  | 256     | Model width. Bigger = more capacity |
| `N_HEADS`    | 4       | Attention heads. Must divide `EMBED_DIM` |
| `N_LAYERS`   | 6       | Depth. More layers = more expressive |
| `STEPS`      | 5000    | Training steps. More = lower loss (to a point) |
| `LR`         | 3e-4    | Learning rate. Too high = unstable, too low = slow |

---

## How This Scales to Real LLMs

| Aspect         | This model       | GPT-3            |
|----------------|------------------|------------------|
| Params         | ~6.5M            | 175B             |
| Layers         | 6                | 96               |
| Embed dim      | 256              | 12,288           |
| Context        | 128 tokens       | 2,048 tokens     |
| Tokenizer      | char-level (65)  | BPE (50,257)     |
| Training data  | 1MB Shakespeare  | ~300B tokens     |

Same architecture. Just scale.
