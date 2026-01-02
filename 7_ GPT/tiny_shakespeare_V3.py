import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)

# read in text
with open('../tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

# here are all the unique characters that occur in this text
chars = sorted(list(set(text))) # ->  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
vocab_size = len(chars) # vocab_size is 65

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# let's now encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)

# Let's now split up the data into train and validation sets
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel? was 32
block_size = 256 # what is the maximum context length for predictions? was 8
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"device is: {device}")
eval_iters = 200
n_embed = 384
n_head = 6
n_layer = 6
dropout = 0.2

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:block_size+1+i] for i in ix])

    # Move the tensors to the device (CPU or CUDA)
    x, y = x.to(device), y.to(device)

    return x, y

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
#print(xb)
print('targets:')
print(yb.shape)
#print(yb)

for b in range(batch_size): # batch
    for t in range(block_size): # time
        context = xb[b, :t+1]
        target = yb[b, t]
        # print(f"when input is {context}, the target is: {target}")

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # randomly drop neurons every once and a while to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)
        # compute attention scores ("affinities")
        # the variance of wei grows proportionally to the size of the attention heads
        # if Q and K are size 16, the variance of wei will be 16 here.
        # so we have to normalize wei afterwards, or we will have extreme values
        # that softmax won't process well (gradients disappear, attention only on one position, etc...)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)

        # taking the running mean of previous positions can be done quickly by using
        # the lower triangular matrix technique:
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # residual connections:
        # gradients get weaker as they propagate through deep NNs
        # use addition to preserve and distribute gradients back to the input
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # (65, 32) -> "meaning" of each character in our 65 character vocabulary
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) 

        # Token embeddings have no notion of position, so "Hello" == "oHell".
        # We use the position_embedding_table which is (8, 32) to represent the meaning of being 1st, 2nd, etc...
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        #self.sa_head = MultiHeadAttention(4, n_embed//4) # 4 heads of 8 dimensions, final results will be concatenated back to 32

        # attention is "gathering data"
        # feed forward is "thinking on data"
        # self.ffwd = FeedForward(n_embed)

        # attention + feed forward now done by blocks
        self.blocks = self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)

        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers

        # idx = Indices of Characters -> (32, 8) during training, (1, 8) during generation
        # ex: [[24, 43, 58, 13, 1, 10, 12, 60], ...]

        #print("BigramLanguageModel -> forward()")
        #print(f"idx.shape = {idx.shape}")
        B, T = idx.shape

        # for each index in idx, lookup its 32 number "meaning", repeat for each character in batch
        # tok_emb.shape = (32, 8, 32) during training
        tok_emb = self.token_embedding_table(idx)

        # pos_emb starts as [0, 1, 2, 3, 4, 5, 6, 7]
        # then we grab the 32 number "meaning" of each position
        # final result is (8, 32)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        # pos_emb gets broadcasted: pos_emb.shape = (8, 32) -> (1, 8, 32) -> copied 32 times -> (32, 8, 32)
        # each batch (1, 8, 32) gets the same position embeddings added to their different token embeddings.
        x = tok_emb + pos_emb

        #x = self.sa_head(x) # apply one head of self-attention

        x = self.blocks(x)

        # refresher: logits are scores for the possible next character
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            #print(logits.shape)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            # PyTorch's cross_entropy() really wants C to be in the second dimension so we have to change shapes
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # Ensure data is on the same device as the model
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = BigramLanguageModel()
m = model.to(device)

# torch.optim.SGD for gradient descent
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

print(f"Entering training loop.")
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=100)[0].tolist()))