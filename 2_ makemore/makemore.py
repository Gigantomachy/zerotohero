import torch
import matplotlib.pyplot as plt

words = open("names.txt", "r").read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# build two character count matrix

N = torch.zeros((27, 27), dtype=torch.int32)

for w in words:
	chs = ['.'] + list(w) + ['.']
	for ch1, ch2 in zip(chs, chs[1:]):
		ix1 = stoi[ch1]
		ix2 = stoi[ch2]
		N[ix1, ix2] += 1

# code for the plot

plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')

for i in range(27):
	for j in range(27):
		chstr = itos[i] + itos[j]
		plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
		plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')

#plt.show()

# code for generation

g = torch.Generator().manual_seed(2147483647)

P = N.float()
Ps = P.sum(1, keepdims=True) # reduce along the columns, collapsing the dimension

print(Ps.shape)

# experiment
print(Ps)
print(P.sum(0, keepdims=True))
# ----------

P = P / Ps # P is now a row vector of probabilities for ".a", ".b", etc...
# we really should be using in place operations here /=

print(P.shape)

for i in range(10):
	out = []
	ix = 0 # ix = 0 because that is the '.' character - start of sequence
	while True:
		p = P[ix]
		ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
		out.append(itos[ix])
		if ix == 0:
			break
	print(''.join(out))

# evaluate the model

for w in words[:3]:
	chs = ['.'] + list(w) + ['.']
	for ch1, ch2 in zip(chs, chs[1:]):
		ix1 = stoi[ch1]
		ix2 = stoi[ch2]
		prob = P[ix1, ix2]
		print(f'{ch1}{ch2}: {prob:.4f}')

