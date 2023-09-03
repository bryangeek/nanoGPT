"""
Playground for the Shakespear dataset tokenized by chars.
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

device = ('cuda' if torch.cuda.is_available()
          else 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
          else 'cpu')
print(device)
print(torch.backends)

dataset='shakespeare_char'

# Load the data file.
data_dir = os.path.join('data', dataset)
input_file_path = os.path.join(data_dir, 'input.txt')
with open(input_file_path, 'r') as f:
    data_text = f.read()
print(f"length of dataset in characters: {len(data_text):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data_text)))
vocab_size = len(chars)
print(f"all the unique characters: {''.join(chars)}")
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Encode the dataset into a tensor
data = torch.tensor(encode(data_text), dtype=torch.long)

# create the train and test splits
n = len(data)
break_point = int(n*0.9)
train_data = data[:break_point]
val_data   = data[break_point:]


batch_size = 4
block_size = 8
print(train_data[:block_size+1])

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i  :i+block_size  ] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
print(xb)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            # print("Going to make loss")
            # print(f"logits 0 = {logits[0]}")
            # print(f"targets 0 = {targets[0]}")
            loss = F.cross_entropy(logits, targets)
            # print(logits.shape, targets.shape, loss.shape)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get the predictions
            logits, loss = self(idx)
            # Focus on the last time step
            logits = logits[:, -1, :]  # Becomes B, C
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)  # B, C
            # Sample from the dist
            idx_next = torch.multinomial(probs, num_samples=1) # B, 1
            # Append sampled index to the sequence
            idx = torch.cat((idx, idx_next), dim=1) # B, T+1
        return idx


m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(f"Num params: {len(list(m.parameters()))}")
print("Loss:")
print(loss)

start = torch.zeros((1, 1), dtype=torch.long)  # zero is the new line char.
output = m.generate(start, max_new_tokens=100)[0].tolist()
print("Untrained model predictions with the prompt of a newline char:")
print(decode(output))
print('\n\n')

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
batch_size = 32
for steps in range(8000):
    # Sample a batch of data
    xb, yb = get_batch('train')
    # Evaluate the loss.
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)

    # Backward pass.
    # for p in m.parameters():
    #    p.grad = None
    loss.backward()
    optimizer.step()

    # Update params
    #for p in m.parameters():
    #    p.data += -0.01 * p.grad

print(loss)

print("Trained model predictions with the prompt of a newline char:")
output = m.generate(start, max_new_tokens=500)[0].tolist()
print(decode(output))
print('\n\n')
