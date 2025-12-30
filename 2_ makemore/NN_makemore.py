import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# training set of bigrams (x, y)
# e.g ".emma + .ava" would be
# xs = [0,  4,  12, 12, 0,  0,  21, 0]   '.', 'e', 'm', 'm', 'a', '.', 'a', 'v'
# ys = [4,  12, 12, 0,  0,  21, 0,  0]   'e', 'm', 'm', 'a', '.', 'a', 'v', '.'
# where ys is the label - or outputs that we expect given an input from xs

xs, ys = [], []
for w in words:
	chs = ['.'] + list(w) + ['.']
	for ch1, ch2 in zip(chs, chs[1:]):
		ix1 = stoi[ch1]
		ix2 = stoi[ch2]
		xs.append(ix1)
		ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

print(xs)
print(ys)

xenc = F.one_hot(xs, num_classes=27).float()

print(xenc.shape)
print(xenc)

# randomly initialize 27 neurons' weights. each neuron receives 27 inputs
# columns of W are the neurons, rows are the weights of a neuron -> likelihood of a character given an input
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

for k in range(100):

    # forward pass
    logits = xenc @ W
    counts = logits.exp()
    prob = counts / counts.sum(1, keepdim=True)
    loss = -prob[torch.arange(ys.shape[0]), ys].log().mean() + 0.01*(W**2).mean()

    # xenc is a series of one-hot tensors, (5, 27) if we use ".emma"
    # since xenc is a bunch of [0, 0, ... X, ... 0, 0], xenc @ W actually just pulls out rows from W
    # logits here would be (5, 27), each row corresponds to an input character, and the columns of each row
    # represent the NN's probability distribution for the 2nd character
    # that is why we want 27 neurons -> why W is (27, 27). If W were (27, 1), we would have (5, 1) -> not a probability distribution
    # if W were (27, 29) -> logits = (5, 29) -> 2 extra meaningless parameters since we only have 27 characters

    # right now, logits is a bunch of negative and positive numbersm so can't be counts. let's interpret these to be "log counts"
    # to get counts from log counts, we exponentiate. Then we can turn these "counts" into a probability.
    # process is called "softmax"

    #print(f'{logits.shape=}')
    #print(f'{counts.shape=}')
    #print(f'{prob.shape=}')
    #print(f'{loss.shape=}')

    if k % 10 == 0:
        print(f'Step {k}: loss = {loss.item():.4f}')

    W.grad = None # zero the gradients
    loss.backward() # backprop to set the gradients

    W.data += -10 * W.grad # 10 seems big but actually works well here

g.manual_seed(2147483647)

# test our NN by sampling
for k in range(20):

    out = []
    ix = 0
    while True:
        ixenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()

        logit = ixenc @ W
        counts = logit.exp()
        prob = counts / counts.sum(1, keepdim=True)

        #this selects an index -> sample from distribution, tell you which index was selected
        ix = torch.multinomial(prob, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

    print(''.join(out))

# loss = -prob[torch.arange(5), ys].log() is equivalent to:

# nlls = torch.zeros(5)

# # assuming we are still dealing with ".emma"
# for i in range(5):
# 	# i-th bigram:
# 	x = xs[i].item() # input character index
# 	y = ys[i].item() # label character index
			
# 	p = probs[i, y]
# 	logp = torch.log(p)

# 	nll = -logp
# 	nlls[i] = nll


