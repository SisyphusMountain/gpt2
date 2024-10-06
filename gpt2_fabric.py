from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import math
import time
import os
import logging
import argparse
# additional imports from PyTorch and Lightning
import lightning as L
from torch.utils.data import IterableDataset, DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"
disable_compilation=False


# Set up argument parser
parser = argparse.ArgumentParser(description="A script to demonstrate logging level control.")
parser.add_argument('--log', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
                    help="Set the logging level (default: info)")

# Parse arguments
args = parser.parse_args()

# Set logging level based on user input
log_level = getattr(logging, args.log.upper(), logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')




logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# Try to create a dataset class that will output any given token.
class GPTDataset(IterableDataset):
    def __init__(self, B, T, split):
        assert split in {"train", "val"}
        self.B = B
        self.T = T
        self.split = split
        self.data_root = "edu_fineweb10B"
        shards = os.listdir(self.data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(self.data_root, s) for s in shards]
        self.shards = shards
        self.current_position = 0
    def __iter__(self):
        shard_iter = iter(self.shards)
        for shard_path in shard_iter:
            tokens = load_tokens(shard_path)
            current_position = 0
            while current_position + self.B * self.T + 1 < len(tokens):
                buf = tokens[current_position:current_position + self.B * self.T + 1]
                current_position += self.B * self.T
                yield buf


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

logging.info("Check if pinning the memory is OK.")
logging.debug("""Choosing batch_size=None should allow us to not add a batching dimension to
                the data, which is already batched in the dataset.""")

B = 16
if B == 20:
    total_batch_size = 512000
elif B == 16:
    total_batch_size = 524288
elif B == 24:
    total_batch_size = 589824
else:
    raise ValueError("Batch size not supported")
T = 1024
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)


# When using the commented dataloader instead of the uncommented train_dataloader, the performance drops to the level of uncompiled code.
train_dataloader = DataLoader(GPTDataset(B, T, "train"),
                              batch_size=None,
                              shuffle=False,
                              pin_memory=True,
                              pin_memory_device="cuda:0")

# train_dataloader = DataLoader(GPTDataset(B, 1024, "train"), batch_size=None, shuffle=False)

# val_dataloader = DataLoader(GPTDataset(B, 1024, "val"), batch_size=None, shuffle=False)
model = GPT(GPTConfig(vocab_size=50304, block_size=1024))

model.to(device)
model.train()
logging.info(f"used cuda memory after creating model: {torch.cuda.memory_allocated()}")
optimizer = model.configure_optimizers(weight_decay=0.1,
                                       learning_rate=6e-4)
torch.set_float32_matmul_precision("high")



logging.info(f"total desired batch size: {total_batch_size}")
logging.info(f"=> computed gradient accumulation steps {grad_accum_steps}")

max_coefficient = 1.0 # Multiplicative coefficient for the learning rate schedule
min_coefficient = max_coefficient  * 0.1
warmup_steps = 715
max_steps = 19500

def get_scheduler(max_coefficient, min_coefficient, warmup_steps, max_steps):
    def get_lr(step):
        if step < warmup_steps:
            return max_coefficient * (step+1)/(warmup_steps)
        elif step > max_steps:
            return min_coefficient
        else:
            decay_ratio = (step - warmup_steps)/(max_steps - warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_coefficient + coeff * (max_coefficient - min_coefficient)
    return get_lr
@torch.compile(disable=disable_compilation)
def optim_step():
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

@torch.compile(mode="max-autotune", disable=disable_compilation)
def training_step(x, y):
    optimizer.zero_grad()
    with torch.autocast(device_type = device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss = loss / grad_accum_steps
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                          max_norm =1.0,
                                          error_if_nonfinite=True)
    logging.debug("maybe the norm error if nonfinite slows down the program")
    return loss, norm

get_lr = get_scheduler(max_coefficient,
                       min_coefficient,
                       warmup_steps,
                       max_steps)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

logging.debug("""Please make sure that we actually want to do a certain number of steps,
                rather than a certain number of epochs.""")

num_epochs = 5
for epoch in range(num_epochs):
    total_loss = torch.tensor(0.0)
    t0 = time.time()
    t1 = time.time()
    for step, buf in enumerate(train_dataloader):
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logging.debug(f"In the training loop, x shape {x.shape}, y shape {y.shape}")
        loss, norm = training_step(x, y)
        total_loss += loss.detach().cpu()
        if step % grad_accum_steps == grad_accum_steps - 1:
            optim_step()
            logging.info(f"Epoch {epoch}, step {step}, loss {total_loss.item()}")
            t0 = t1
            t1 = time.time()
            dt = (t1 - t0)
            tokens_per_second = B * T * grad_accum_steps / dt
            logging.info(f"tokens per second = {tokens_per_second:.0f} | loss: {total_loss.item():.6f} | gradient norm {norm:.4f} | dt {dt*1000:.6f} ms")
            total_loss = torch.tensor(0.0)
