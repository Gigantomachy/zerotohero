import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open("names.txt", "r").read().splitlines()

random.seed(2147483647)
random.shuffle(words)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


block_size = 3 # context length: how many characters to use to predict the next one
def build_dataset(words):
    X, Y = [], [] # once again, X is our input, and Y is our label (what we want the output to be)
    #for w in words[:5]:
    for w in words:
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
    print(X.shape, Y.shape)
    return X, Y

n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtra, Ytra = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xtst, Ytst = build_dataset(words[n2:])

num_examples = Xtra.shape[0] # number of 3 character inputs in our training data set

g = torch.Generator().manual_seed(2147483647) # for easier debugging

# Setting up parameters
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


batch_size = 32
for i in range(10000):
    ix = torch.randint(0, num_examples, (batch_size,)) # IMPORTANT: do this in the loop, so we don't select the same data points over and over

    #ix is an array of 32 random ints, Xtra[ix] is (32, 3) corresponding to those random ints, Ytra[ix] is the respective label
    emb = C[Xtra[ix]] # (32, 3, 2) if we are working with the 5 first names still

    h = torch.tanh(emb.view(batch_size, 6) @ W1 + b1) # (32, 27)
    logits = h @ W2 + b2 # -> 32, 27 so shape looks good
    loss = F.cross_entropy(logits, Ytra[ix])

    # F.cross_entropy replaces these lines:
    # counts = logits.exp()
    # prob = counts / counts.sum(1, keepdims=True)
    # loss = -prob[torch.arange(32), Y].log().mean()

    for p in parameters:
        p.grad = None
    loss.backward()

    for p in parameters:
        p.data += -0.1 * p.grad

    if i % 9999 == 0:
        print(loss)

# print out the model's performance with the dev training set
emb = C[Xdev]
h = torch.tanh(emb.view(-1, 6) @ W1 + b1) 
logits = h @ W2 + b2
loss = F.cross_entropy(logits, Ydev)

print(loss)

# sample from the model

for _ in range(20):
    out = []
    context = [0] * block_size

    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1) 
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix] 
        out.append(ix)

        if ix == 0:
            break

    print(''.join(itos[i] for i in out))
