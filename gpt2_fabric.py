from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os
# additional imports from PyTorch and Lightning
import lightning as L
from torch.utils.data import Dataset, DataLoader

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# Try to create a dataset class that will output any given token.
class GPTDataset(Dataset):
    def __init__(self, B, T, split):
        assert split in {"train", "val"}
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        self.current_shard = load_tokens(self.shards[self.current_shard])
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0
    def __iter__(self):
        shard_iter = iter(self.tokens)
        for shard_path in shard_iter:
            tokens = load_tokens(shard_path)
            current_position = 0
            while current_position + self.B * self.T + 1 < len(tokens):
                buf = tokens[current_position:current_position + self.B * self.T + 1]
                x = buf[:-1].view(self.B, self.T)
                y = buf[1:].view(self.B, self.T)
                current_position += self.B * self.T
                yield x, y

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)))
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2)

        # att = torch.einsum("bhxc, bhyc -> bhxy", q, k)*(1.0/np.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = torch.einsum("bhax, bhxc -> bhac", att, v)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
    

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe= nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight # weight sharing
        self.apply(self._init_weights)
        # TEST: compile the forward method
        self.forward = torch.compile(self.forward, mode="max-autotune")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2*self.config.n_layer)**-0.5
            
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # We can modify that to take into account the size of the layers more precisely.
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def configure_optimizers(self,
                             weight_decay,
                             learning_rate,
                             ):
        param_dict = {pn: p for (pn, p) in self.named_parameters() if p.requires_grad}
        decay_params = [p for (n, p) in param_dict.items() if p.dim() >= 2]
        non_decay_params = [p for (n, p) in param_dict.items() if p.dim() < 2]
        optim_groups = [{"params": decay_params, "weight_decay": weight_decay},
                        {"params": non_decay_params, "weight_decay": 0.0}]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay_params = sum(p.numel() for p in non_decay_params)
        print(f"Number of decayed parameter tensors {len(decay_params)} for a total of {num_decay_params} parameters")
        print(f"Number of non-decayed parameter tensors {len(non_decay_params)} for a total of {num_nondecay_params} parameters")
        # pass the learning rate as a tensor, because apparently this is needed afterwards to compile the optimizer with the scheduler.
        optimizer = torch.optim.AdamW(optim_groups, lr=torch.tensor(learning_rate), betas=(0.9, 0.95), eps=1e-8, fused=True)
        return optimizer

    def forward(self, idx, targets=None):
        """
        input:
            idx: (B, T)
        """
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(T, device=idx.device)
        pos_embed = self.transformer.wpe(pos) # shape (T, C)
        tok_embed = self.transformer.wte(idx) # shape (B, T, C)
        x = tok_embed + pos_embed # By broadcasting, we obtain shape (B, T, C). The same pos_embed tensor of shape (T, C) is added to each batch.

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # logits has shape (B, T, V): probability distribution over the vocabulary (V) for each token (T) in the sequence for each batch (B).
        if targets is not None:
            # Turn the dimensions (B, T, V) into (B*T, V) and (B, T) into (B*T) to compute the cross-entropy loss.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # cross-entropy takes unnormalized logits and ground truth targets (not one-hot encoded, but just the indices of the correct tokens)
        else:
            loss = None
        return logits, loss
    

