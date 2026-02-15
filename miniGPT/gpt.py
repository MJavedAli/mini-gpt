import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.manual_seed(42)

# ── Wikipedia Article Corpus ──────────────────────────────────────────
# Place a plain-text Wikipedia article in the same directory named 'wiki_article.txt'
# with open('wiki_article.txt', 'r', encoding='utf-8') as f:
with open('tafs.txt', 'r', encoding='utf-8') as f:

    full_text = f.read()

# Truncate to first 100 KB for manageable training time
text = full_text[:100_000]

# Build character-level vocabulary
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

# ── Hyperparameters ───────────────────────────────────────────────────
block_size    = 32      # larger context window
batch_size    = 32
max_iters     = 8000    # more epochs for richer text
eval_interval = 500
lr            = 3e-3
n_embd        = 64      # increased embedding dimension
n_head        = 4       # keep manageable for CPU/GPU
n_layer       = 4       # deeper model for complexity

# ── Data batching ─────────────────────────────────────────────────────
def get_batch():
    idx = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size]     for i in idx])
    y = torch.stack([data[i+1:i+1 + block_size] for i in idx])
    return x, y

# ── LayerNorm, Attention, FeedForward, Blocks ─────────────────────────
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))
        self.eps   = eps

    def forward(self, x):
        mu  = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head   = n_head
        self.head_dim = n_embd // n_head
        self.q_proj   = nn.Linear(n_embd, n_embd)
        self.k_proj   = nn.Linear(n_embd, n_embd)
        self.v_proj   = nn.Linear(n_embd, n_embd)
        self.out      = nn.Linear(n_embd, n_embd)
        self.dropout  = nn.Dropout(0.1)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
                 .view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.out(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # switched from ReLU
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)


class GPTBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1  = LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head)
        self.ln2  = LayerNorm(n_embd)
        self.ff   = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb   = nn.Embedding(block_size, n_embd)
        self.blocks    = nn.Sequential(*[GPTBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f      = LayerNorm(n_embd)
        self.head      = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x   = tok + pos
        x   = self.blocks(x)
        x   = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)[:, -1, :] / temperature
    
            if top_k > 0:
                v, ix = torch.topk(logits, top_k)
                probs = torch.zeros_like(logits).scatter(1, ix, F.softmax(v, dim=-1))
            else:
                probs = F.softmax(logits, dim=-1)
    
            next_t = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_t], dim=1)
        return idx


# ── Training Loop ──────────────────────────────────────────────────────
model = GPT()
optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)


for it in range(max_iters):
    xb, yb = get_batch()
    _, loss = model(xb, yb)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if it % eval_interval == 0:
        print(f"Iter {it:4d} | loss {loss.item():.4f}")

# ── Text Generation ────────────────────────────────────────────────────

# seed_text = "Humans and machines can"  # All lowercase
seed_text = "Humans and machines can"  # All lowercase

# Convert to tokens only if characters are in vocab
start = torch.tensor([[stoi[ch] for ch in seed_text.lower()]], dtype=torch.long)

# Run generation
output = model.generate(start, max_new_tokens=300, temperature=1.2, top_k=30)[0].tolist()
print("\n>>> Generated text:\n", "".join(itos[i] for i in output))
