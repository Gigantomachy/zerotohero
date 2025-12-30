import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures

# read in all the words
words = open('../names.txt', 'r').read().splitlines()
print(len(words))
print(max(len(w) for w in words))
print(words[:8])

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)
print(itos)
print(vocab_size)

# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%

# utility function we will use later when comparing manual gradients to PyTorch gradients
def cmp(s, dt, t):
  ex = torch.all(dt == t.grad).item()
  app = torch.allclose(dt, t.grad)
  maxdiff = (dt - t.grad).abs().max().item()
  print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 64 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1 # using b1 just for fun, it's useless because of BN
# Layer 2
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

# Note: I am initializating many of these parameters in non-standard ways
# because sometimes initializating with e.g. all zeros could mask an incorrect
# implementation of the backward pass.

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
  p.requires_grad = True

batch_size = 32
n = batch_size # a shorter variable also, for convenience
# construct a minibatch
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

# forward pass, "chunkated" into smaller steps that are possible to backward one at a time

emb = C[Xb] # embed the characters into vectors
embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
# Linear layer 1
hprebn = embcat @ W1 + b1 # hidden layer pre-activation
# BatchNorm layer
bnmeani = 1/n*hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
# Non-linearity
h = torch.tanh(hpreact) # hidden layer
# Linear layer 2
logits = h @ W2 + b2 # output layer
# cross entropy loss (same as F.cross_entropy(logits, Yb))
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes # subtract max for numerical stability
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

print("Yb.shape, logprobs.shape, logprobs[range(n), Yb].shape")
print(Yb.shape)
print(logprobs.shape)
print(logprobs[range(n), Yb].shape) #
# Yb should be an array of 32

# PyTorch backward pass
for p in parameters:
  p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv, # afaik there is no cleaner way
          norm_logits, logit_maxes, logits, h, hpreact, bnraw,
         bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani,
         embcat, emb]:
  t.retain_grad()
loss.backward()
print(loss)


# Exercise 1: backprop through the whole thing manually, 
# backpropagating through exactly all of the variables 
# as they are defined in the forward pass above, one by one

#loss = -logprobs[range(n), Yb].mean()
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(32), Yb] = 1.0 * (-1.0 / 32.0)

cmp('logprobs', dlogprobs, logprobs)

#logprobs = probs.log()
dprobs = dlogprobs * (1.0 / probs)

cmp('probs', dprobs, probs)

#probs = counts * counts_sum_inv
#dcounts = dprobs * counts_sum_inv
#cmp('counts', dcounts, counts)

# leave counts for now because it is used twice

# NOTE: counts_sum_inv was broadcasted horizontally 27 times to be multiplied with counts.
	
# d(broadcast(counts_sum_inv)) = dprobs * counts
	
# probs = counts * counts_sum_inv
# probs = [c_1 * counts_sum_inv, c_2 * counts_sum_inv, c_3 * counts_sum_inv, etc...]
# probs_i = c_i * counts_sum_inv
	
# where each c_i is (32, 1), a column for 32 inputs, the counts for one character to come next for that input
	
# The multivariate chain rule, if multiple variables depend on one variable: dL/ds = dL/dp_1 * dp_1/ds + dL/dp_2 * dp_2/ds + ...
	
# dloss/dcounts_sum_inv = dloss/dprobs_1 * dprobs_1/dcounts_sum_inv + dloss/dprobs_2 * dprobs_2/dcounts_sum_inv
	
# of course, if we have a multiplication, the derivative is simply the coefficient, so d(probs_i)/d(counts_sum_inv) = c_i
	
# dloss/dcounts_sum_inv = dloss/dprobs_1 * c_1 + dloss/dprobs_2 * c_2 + ...
	
# Therefore, dcounts_sum_inv = (dprobs * counts).sum(1, keepdim=True)

dcounts_sum_inv = (dprobs * counts).sum(1, keepdim=True)
cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv)

# counts_sum_inv = counts_sum**-1
# simple, use the power rule
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
cmp('counts_sum', dcounts_sum, counts_sum)

# now we can solve for the two counts
# probs = counts * counts_sum_inv
# counts_sum = counts.sum(1, keepdims=True)

# counts_sum = c_1 + c_2 + c_3 + ...
	
# where each c_i is a column with 32 rows from counts.
	
# dloss/d(counts_2) = dloss/d(counts_sum) * d(counts_sum)/d(counts_2)
# dcounts_2 = dcounts_sum * d(counts_sum)/d(counts_2)
	
# d(counts_sum)/d(counts_2) = d(c_1 + c_2 + c_3 + ...)/d(counts_2)
	
# Each c_i is an independent variable. So we are interested in how the loss changes if each individual column changed.
	
# Remember, the gradient of a variable must always have the exact same shape as the variable itself.
	
# Therefore, we treat this as 27 individual derivatives (partial derivatives). In the case of partial derivatives, we treat
# the other variables as constants. So each of these partial derivatives is just 1.
	
# dc_1/dc_1 + dc_2/dc_1 + dc_3/dc_1 + ...
# 	1     +		0	  +		0	  + ...
	
# Repeat for all 27 columns.

dcounts_1 = dprobs * counts_sum_inv
dcounts_2 = torch.ones_like(counts) * dcounts_sum
	
dcounts = dcounts_1 + dcounts_2

cmp('counts', dcounts, counts)

# now dnorm_logits, counts = norm_logits.exp()
# .exp() is e^x, the derivative of e^x is just e^x
# dnorm_logits = dcounts * d(counts)/d(norm_logits)
#                           norm_logits.exp()
#                           counts
# so actually dnorm_logits = dcounts * counts

dnorm_logits = dcounts * counts
cmp('norm_logits', dnorm_logits, norm_logits)

#print(dnorm_logits.shape)
#print(logit_maxes.shape)
#print(logits.shape)

# also nice and easy, same as counts_sum_inv broadcasting above
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
cmp('logit_maxes', dlogit_maxes, logit_maxes)

# now we have dlogits which is actually harder
# logit_maxes = logits.max(1, keepdim=True).values
# norm_logits = logits - logit_maxes # subtract max for numerical stability

dlogits_1 = dnorm_logits
dlogits_2 = F.one_hot(logits.argmax(1), num_classes=logits.shape[1]) * dlogit_maxes

# above we create 32 rows, each with a 27 one-hot tensor, where the hot is where the maximum occured -> local gradient
# we then multiply the local gradient with the upstream gradient which is dlogit_maxes

# print(logits.shape)
# print(logit_maxes.shape)

dlogits = dlogits_1 + dlogits_2
cmp('logits', dlogits, logits)

# moving onto the matrix multiplications
print(logits.shape, h.shape, W2.shape, b2.shape)

# The rules for matrix derivative: Y = X @ W

# dX = dY @ transpose(W)
# dW = transpose(X) @ Y

# (chain rule already taken into account in above)

# Although an easier way to remember all this is that the dimensions must line up.

dh = dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)

cmp('h', dh, h)
cmp('W2', dW2, W2)
cmp('b2', db2, b2)