import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# context length: how many characters to use to predict the next one
block_size = 3
X, Y = [], [] # once again, X is our input, and Y is our label (what we want the output to be)

for w in words[:5]:
#for w in words:
    #print(w)
    context = [0] * block_size # start with a padded context -> so "..." in our case
    for ch in w + '.':
        ix = stoi[ch]  # Convert character to index
        X.append(context) # first loop, this is 000 or ...
        Y.append(ix) # first loop, this is 'e' for "emma"

        #print(''.join(itos[i] for i in context), '——>', itos[ix])

        context = context[1:] + [ix] # slide the context along

X = torch.tensor(X) # 32, 3 -> first 5 words
Y = torch.tensor(Y) # 32,

g = torch.Generator().manual_seed(2147483647) # for easier debugging

# parameters
C = torch.randn((27, 2), generator=g)

# 100 neurons, each neuron currently has 3 inputs x 2 embeddings per input = 6 weights
W1 = torch.randn((6, 100), generator=g) # samples numbers from normal distribution, <-3.0 and >3.0 are rare
b1 = torch.randn(100, generator=g) # 100 biases for 100 neurons

# for softmax layer
W2 = torch.randn((100, 27), generator=g)
b2 = torch.randn(27, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad_(True)

for i in range(10):
    emb = C[X] # (32, 3, 2) if we are working with the 5 first names still

    h = torch.tanh(emb.view(32, 6) @ W1 + b1) # (32, 27)
    logits = h @ W2 + b2 # -> 32, 27 so shape looks good
    loss = F.cross_entropy(logits, Y)

    # F.cross_entropy replaces these lines:
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdims=True)
    # loss = -prob[torch.arange(32), Y].log().mean()

    for p in parameters:
        p.grad = None
    loss.backward()

    for p in parameters:
        p.data += -0.1 * p.grad

    print(loss)