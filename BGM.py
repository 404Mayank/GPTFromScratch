import torch
import torch.nn as nn
from torch.nn import functional as f

n_embed = 32
max_new_tokens = 500
split_ratio = 0.9
batch_size = 64
block_size = 8
epoches = 10000
learning_rate = 1e-2
eval_iters = 300
eval_interval = 300
device = "cuda" if torch.cuda.is_available() else "cpu"

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
input = "input.txt"
with open(input, "r", encoding="utf-8") as file:
    text = file.read()

chars = sorted(set(text))
vocab_size = len(chars)

# Dicts to map char->int and int->char
charToInt = {character: integer for integer, character in enumerate(chars)}
intToChar = {integer: character for integer, character in enumerate(chars)}


# input s,l ; for every c/i in s/l do give coressponding value from dict.
def encode(s):
    return [charToInt[c] for c in s]


def decode(l):
    return "".join(intToChar[i] for i in l)


data = torch.tensor(encode(text), dtype=torch.long)
split_at = int(split_ratio * len(text))  # 90:10 split
train_data = data[:split_at]
val_data = data[split_at:]


def get_batch(split):
    data = train_data if split == "train" else val_data
    xi = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in xi])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in xi])
    x, y = x.to(device), y.to(device)
    return x, y


class BLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        logits = self.lm_head(tok_emb)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = f.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = f.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BLM()
m = model.to(device)


@torch.no_grad()
def est_loss():
    out = {}
    model.eval()
    for splits in ["train", "eval"]:
        losses = torch.zeros(eval_iters)
        for j in range(eval_iters):
            x, y = get_batch(splits)
            logits, loss = model(x, y)
            losses[j] = loss.item()
        out[splits] = losses.mean()
    model.train()
    return out


optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for steps in range(epoches):
    # sample
    xb, yb = get_batch("train")
    # eval
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if steps % eval_interval == 0:
        losses = est_loss()
print("Training Complete")

# generate
idx = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = m.generate(idx, max_new_tokens)[0].tolist()
print((decode(generated)))
