from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tiktoken
enc = tiktoken.get_encoding("gpt2")
import itertools
import math
import time
import os
import logging
import argparse
import psutil
from torch.utils.data import IterableDataset, DataLoader
import wandb
from collections import deque

torch.set_float32_matmul_precision("high")

# Initialize wandb
wandb.init(project='gpt2', dir="./saved_files")

parser = argparse.ArgumentParser()
parser.add_argument('--log', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
                    help="Set the logging level (default: info)")
args = parser.parse_args()
log_level = getattr(logging, args.log.upper(), logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

device = "cuda:0" if torch.cuda.is_available() else "cpu"

disable_compilation = True
if disable_compilation:
    logging.warning("Disabling compilation")
else:
    logging.info("Enabling compilation")

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

# Dataset class
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
        self.worker_index = 0
        self.num_workers = 1
    def __iter__(self):
        self.shards = self.shards[self.worker_index::self.num_workers]
        shard_iter = itertools.cycle(self.shards)
        for shard_path in shard_iter:
            logging.info(f"Loading shard {shard_path}")
            tokens = load_tokens(shard_path)
            current_position = 0
            while current_position + self.B * self.T + 1 < len(tokens):
                buf = tokens[current_position:current_position + self.B * self.T + 1]
                current_position += self.B * self.T
                yield buf
class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T
        return x, y
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
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight  # weight sharing
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, weight_decay, learning_rate):
        param_dict = {pn: p for (pn, p) in self.named_parameters() if p.requires_grad}
        decay_params = [p for (n, p) in param_dict.items() if p.dim() >= 2]
        non_decay_params = [p for (n, p) in param_dict.items() if p.dim() < 2]
        optim_groups = [{"params": decay_params, "weight_decay": weight_decay},
                        {"params": non_decay_params, "weight_decay": 0.0}]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=False)
        return optimizer

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(T, device=idx.device)
        pos_embed = self.transformer.wpe(pos)
        tok_embed = self.transformer.wte(idx)
        x = tok_embed + pos_embed

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            loss = None
        return logits, loss

# Hyperparameters
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
grad_accum_steps = B*T // (B * T)

steps_save = 10000
max_steps = 19500 * grad_accum_steps * 32
batched_steps = max_steps // grad_accum_steps

# Initialize model
config = GPTConfig(vocab_size=50304, block_size=1024)
model = GPT(config)
if not disable_compilation:
    model = torch.compile(model, mode="max-autotune")
model.to(device)

# Optimizer and Scheduler
lr = 6e-4/32
weight_decay = 0.1
max_coefficient = 1.0
min_coefficient = max_coefficient * 0.1
warmup_steps = 731*32

optimizer = model.configure_optimizers(weight_decay, lr)

def get_scheduler(optimizer):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return max_coefficient * (current_step + 1) / (warmup_steps)
        elif current_step > batched_steps:
            return min_coefficient
        else:
            decay_ratio = (current_step - warmup_steps) / (batched_steps - warmup_steps)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_coefficient + coeff * (max_coefficient - min_coefficient)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler

scheduler = get_scheduler(optimizer)





# Training Loop
batch_accum_counter = 0
total_loss = torch.tensor(0.0)
total_tokens_processed = 0
start_time = time.time()
current_step = 0
dataloader = DataLoaderLite(B, T, "train")
for batch_idx in range(max_steps):
    model.train()
    x, y = dataloader.next_batch()
    x = x.to(device)
    y = y.to(device)

    with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
        logits, loss = model(x, y)

    loss = loss / grad_accum_steps
    total_loss += loss.detach().cpu()
    batch_accum_counter += 1

    loss.backward()
    if batch_accum_counter == grad_accum_steps:
        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        current_step += 1

        # Logging
        lr_current = optimizer.param_groups[0]['lr']
        cuda_memory = torch.cuda.memory_allocated() if "cuda" in device else 0
        cpu_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)  # in GB

        current_time = time.time()
        dt = current_time - start_time
        tokens_per_second = (B * T * grad_accum_steps) / dt
        total_tokens_processed += B * T * grad_accum_steps
        eta = (10**10 - total_tokens_processed) / tokens_per_second if tokens_per_second > 0 else float('inf')

        # Log metrics to wandb
        wandb.log({
            'train/loss': total_loss.item(),
            'train/tokens_per_second': tokens_per_second,
            'step': current_step,
        }, step=current_step)

        # Reset counters
        total_loss = torch.tensor(0.0)
        batch_accum_counter = 0
        start_time = time.time()

        # Save checkpoint every steps_save
        if current_step % steps_save == 0:
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint_path = os.path.join('checkpoints', f'checkpoint_step_{current_step}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'current_step': current_step
            }, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")

# Save final model
os.makedirs('checkpoints', exist_ok=True)
final_checkpoint_path = os.path.join('checkpoints', 'final_model.pt')
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'current_step': current_step
}, final_checkpoint_path)
logging.info(f"Saved final model to {final_checkpoint_path}")
