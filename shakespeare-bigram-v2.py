"""
Playground for the Shakespear dataset tokenized by chars.
Version 2 adds attention.
"""
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

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

# We're going to do token embeddings this time.  For this case
# we will use a 32 dimention embedding vector.
n_embed = 32

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


batch_size = 4  # Number of blocks to train at once (to keep GPU busy).
block_size = 8  # Also called max context length.
print(train_data[:block_size+1])

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i  :i+block_size  ] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  # Targets are the chars that follow the input.
    return x, y

xb, yb = get_batch('train')

class Head(nn.Module):
    """Single head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key    = nn.Linear(n_embed, head_size, bias=False)
        self.query  = nn.Linear(n_embed, head_size, bias=False)
        self.value  = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # Compute attention scores using dot product.
        wei = q @ k.transpose(-2, -1) / C**0.5
        # Decode block, so hide the future tokens.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        # Perform the weighted aggregation of the values.
        v = self.value(x)
        out = wei @ v  # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out

class SimpleLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed):
        super().__init__()
        # We are switching away from the Bigram lookup table.
        # Now when given a char, we use a trained table to convert the
        # char into an embedding vector.  In this example going from 65
        # down to 32.  Resulting in a 32 dimensional embedding vector.
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # We are going to add a self-attention head.
        self.sa_head = Head(n_embed)
        # We need a way to map from the embeddings back to the
        # full set of outputs.  We can use a layer of neurons that
        # can be trained for this.
        self.lm_head = nn.Linear(n_embed, vocab_size)
        # Another thing we need to add is a way for the model to know the position
        # of each token in the sequence.  So we're going to add a positional embedding.
        self.pos_embedding_table = nn.Embedding(block_size, n_embed)

    def forward(self, idx, targets=None):
        # First get the token embeddings.
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        # Next get the positional embedding.
        B, T = idx.shape  # We need the number of tokens.
        pos_vec = torch.arange(T)
        pos_emb = self.pos_embedding_table(pos_vec)  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C) + (T, C) --> torch adds B to pos_emb.
        # Now we can feed these embeddings into the attention head.
        x = self.sa_head(x)
        # The log of counts is now based on the results of the attention.
        logits = self.lm_head(x)  # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            # Batch by Time by Chars tesor.  Need to flatten in order
            # to compute all the cross_entropy at once.
            B, T, C = logits.shape
            logits  = logits.view(B*T, C)
            targets = targets.view(B*T)
            # Reminder that cross_entropy is the negative log liklihood. It's
            # a measure of how good the quality of the logits are compared
            # to the targets.  Lower is better.
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop the idx to the last block_size tokens.
            idx_cond = idx[:, -block_size:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus on the last time step
            logits = logits[:, -1, :]  # Becomes B, C
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)  # B, C
            # Sample from the dist to get the next char
            idx_next = torch.multinomial(probs, num_samples=1) # B, 1
            # Append sampled index to the sequence
            idx = torch.cat((idx, idx_next), dim=1) # B, T+1
        return idx


model = SimpleLanguageModel(vocab_size, n_embed)
logits, loss = model(xb, yb)
print(logits.shape)
print(f"Num params: {sum(p.nelement() for p in model.parameters()):,}")
print("Loss:", loss.item())

start = torch.zeros((1, 1), dtype=torch.long)  # zero is the new line char.
output = model.generate(start, max_new_tokens=100)[0].tolist()
print("\n\nUntrained model predictions with the prompt of a newline char:")
print(decode(output))
print('\n\n')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
batch_size = 32
print("Training:")
for step in range(10000):
    # Sample a batch of data
    xb, yb = get_batch('train')
    # Evaluate the loss.
    logits, loss = model(xb, yb)

    # Backward pass.
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()  # Updates params.
    if not step % 1000:
        print(f"Step: {step:4d}: loss: {loss.item()}")

print(f"Final loss: {loss.item()}")

print("\n\nTrained model predictions with the prompt of a newline char:")
output = model.generate(start, max_new_tokens=500)[0].tolist()
print(decode(output))
print('\n\n')

