import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

words = open("../names.txt", "r").read().splitlines()

random.seed(2147483647)
random.shuffle(words)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

vocab_size = 27
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

# MLP revisited
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 200 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g)
b1 = torch.randn(n_hidden,                        generator=g) * 0.01
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.01
b2 = torch.randn(vocab_size,                      generator=g) * 0

parameters = [C, W1, b1, W2, b2]

for p in parameters:
	p.requires_grad = True

