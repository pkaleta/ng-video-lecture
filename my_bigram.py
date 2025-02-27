import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32
n_heads = 4
n_blocks = 4
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """Single head self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # (B, T, C=32)

        # Self-attention
        q = self.query(x)  # (B, T, HS)
        k = self.key(x)  # (B, T, HS)
        wei = q @ k.transpose(-1, -2) * C**-0.5  # (B, T, HS) @ (B, HS, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        # wei = self.dropout(wei)

        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        v = self.value(x)  # (B, T, HS)
        return wei @ v  # (B, T, T) @ (B, T, HS) --> (B, T, HS)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        # self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.query = nn.Linear(n_embd, n_embd, bias=False)  # (C, HS)
        self.key = nn.Linear(n_embd, n_embd, bias=False)  # (C, HS)
        self.value = nn.Linear(n_embd, n_embd, bias=False)  # (C, HS)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        # self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_embd, n_embd)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # (B, T, C)
        q = (
            self.query(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        )  # (B, nh, T, hs)
        k = (
            self.key(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        )  # (B, nh, T, hs)
        att = (
            q @ k.transpose(-1, -2) * C**-0.5
        )  # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        att = att.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, nh, T, T)
        # wei = self.dropout(wei)

        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        v = (
            self.value(x).view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        )  # (B, nh, T, hs)
        y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) --> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        out = self.proj(y)  # (B, T, C)
        # out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffwd(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.sa_heads = MultiHeadAttention(n_heads=n_heads)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))  # apply self-attention, (B, T, C)
        x = x + self.ffwd(self.ln2(x))  # apply linear MLP, (B, T, C)
        return x


# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd=n_embd, n_heads=n_heads) for _ in range(n_blocks)]
        )
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # print(idx.shape)
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embs = self.token_embedding_table(idx)  # (B,T,n_embd)
        pos_embs = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embs + pos_embs
        # print(x.shape)
        x = self.blocks(x)
        x = self.final_ln(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx[:, -block_size:])
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
