# MiniGPT — From-Scratch Transformer (Character-Level)

Reference implementation of a GPT-style decoder-only Transformer built from first principles in PyTorch, demonstrating multi-head self-attention, causal masking, and autoregressive language modeling.

This project demonstrates the internal mechanics of modern Large Language Models (LLMs) — including:
- Token embeddings
- Positional embeddings
- Multi-head self-attention
- Causal masking
- Residual connections
- Pre-LayerNorm architecture
- Autoregressive generation
- Top-k sampling

The goal is not scale, but clarity of architecture.

# Purpose

This repository exists to:
- Deconstruct GPT architecture into understandable components
- Demonstrate attention mechanics at tensor level
- Provide a clean educational reference for engineers
- Enable experimentation with small-scale language modeling

This is not intended to compete with production LLMs.
It is designed to explain them.

# Architecture Overview
This model implements a decoder-only Transformer, similar in structure to GPT-2.

```
Input Tokens
   ↓
Token Embedding
   ↓
Positional Embedding
   ↓
[ Transformer Block × N ]
   ↓
Final LayerNorm
   ↓
Linear Projection → Vocabulary Logits
   ↓
Softmax
   ↓
Next Token Prediction
```

# Core Components
 ## Tokenization
- Character-level vocabulary
- Maps characters → integer IDs
- Suitable for small datasets
- Avoids external tokenizer dependencies
  
### Trade-off:

Harder learning compared to subword tokenization
Simpler implementation

## Embeddings

```
nn.Embedding(vocab_size, n_embd)
```

Each token becomes a dense vector representation.
Positional embeddings inject sequence order information.

## Multi-Head Self-Attention

Implements scaled dot-product attention:

$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V$


### Features:

- Multi-head attention
- Causal masking (no future token leakage)
- Dropout regularization

#### Tensor flow per head:

```
(B, T, C)
→ Q, K, V
→ (B, n_head, T, head_dim)
→ Attention scores (T × T)
→ Weighted sum of values
→ Concatenate heads
```

## Transformer Block

Pre-LayerNorm architecture:

```
x = x + Attention(LN(x))
x = x + FeedForward(LN(x))
```

#### Advantages:

- Improved training stability
- Better gradient flow

## Feedforward Network

Two-layer MLP with GELU activation:

```
Linear → GELU → Linear → Dropout
```

Dimensional expansion factor: 4×

Standard GPT design choice.

## Training Objective

$$
\mathcal{L} = - \log P(x_t \mid x_{\lt t})
$$


Where:

- $x_t$ is the target token at position $t$
- $x_{<t}$ represents all previous tokens
- $P(x_t \mid x_{<t})$ is the predicted probability of the next token

### Hyperparameters (Default)

```
block_size = 32
batch_size = 32
n_embd = 64
n_head = 4
n_layer = 4
max_iters = 2000
```

This results in ~100K–300K parameters depending on vocab size.
Designed for CPU experimentation.

## Installation

#### Create virtual environment
```
python3 -m venv venv
source venv/bin/activate
```
#### Install dependencies
```
python -m pip install torch numpy
```
#### Running the Model

Place your training text in:
wiki_article.txt
Then run:
```
python ManualGPT_wiki.py
```

Training logs will display:

```
Iter 0 | loss ...
Iter 500 | loss ...
...
```

#### After training completes, text generation begins.

Text Generation

#### Uses:

- Temperature scaling
- Top-k sampling
- Autoregressive decoding

Example:

```
model.generate(start, max_new_tokens=300, temperature=1.2, top_k=30)
```
Controls creativity vs determinism.

## Limitations

This implementation is intentionally minimal.

#### Limitations include:

- Character-level tokenization
- Very small parameter count
- Small training corpus
- No distributed training
- No mixed precision
- No checkpointing

This model will memorize small datasets.
It will not exhibit emergent reasoning or deep semantic generalization.
That requires scale.

## What This Demonstrates

Despite limitations, this implementation captures:

- The full architectural pattern of GPT models
- The attention mechanism used in large-scale LLMs
- The training paradigm behind autoregressive models

The difference between this and GPT-4 is scale — not architectural mystery.

## Suggested Extensions

- For engineers looking to expand:
- Switch to subword tokenization (e.g., BPE)
- Increase embedding size and depth
- Add model checkpointing
- Add learning rate scheduler
- Add GPU support
- Visualize attention maps
- Replace character tokenizer with word-level tokenizer
- Train on larger corpus (multi-MB dataset)

## Educational Value

This repository is ideal for:

- Engineers transitioning into AI/ML
- Understanding Transformer internals
- Interview preparation
- Architecture discussions
- Research prototyping

## Final Note

Modern LLMs are not magic.
They are scaled-up versions of:

- Linear layers
- Attention
- Layer normalization
- Gradient descent

Understanding this implementation means understanding the foundation of large-scale generative AI systems.
