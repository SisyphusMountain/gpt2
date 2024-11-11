from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import tiktoken
enc = tiktoken.get_encoding("gpt2")
from hellaswag import render_example, iterate_examples
import itertools
import math
import time
import os
import logging
import argparse
import psutil
from torch.utils.data import IterableDataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from collections import deque

torch.set_float32_matmul_precision("high")

# Initialize wandb logger
wandb_logger = WandbLogger(project='gpt2', save_dir="./saved_files")

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
import os
import random
import itertools
import logging
from torch.utils.data import IterableDataset

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
        # Distribute shards among workers if using multiple workers
        self.shards = self.shards[self.worker_index::self.num_workers]
        shard_iter = itertools.cycle(self.shards)
        
        for shard_path in shard_iter:
            logging.info(f"Loading shard {shard_path}")
            tokens = load_tokens(shard_path)
            indices = list(range(0, len(tokens) - self.B * self.T, self.B * self.T))
            random.shuffle(indices)  # Shuffle indices to yield in random order

            for start_idx in indices:
                buf = tokens[start_idx: start_idx + self.B * self.T + 1]
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

# Define the LightningModule
class GPTLightningModule(pl.LightningModule):
    def __init__(self, config, total_batch_size, B, T, grad_accum_steps, max_steps, disable_compilation=True):
        super().__init__()
        self.model = GPT(config)
        if not disable_compilation:
            self.model = torch.compile(self.model, mode="max-autotune")

        self.automatic_optimization = False  # Disable automatic optimization
        if not self.automatic_optimization:
            logging.warning(f"Automatic optimization is set to {self.automatic_optimization}")

        self.config = config
        self.B = B
        self.T = T
        self.total_batch_size = total_batch_size
        self.grad_accum_steps = grad_accum_steps
        self.max_steps = max_steps
        self.batched_steps = max_steps // grad_accum_steps

        self.lr = 6e-4
        self.weight_decay = 0.1
        # Scheduler parameters
        self.max_coefficient = 1.0
        self.min_coefficient = self.max_coefficient * 0.1
        self.warmup_steps = 731

        # Initialize variables for gradient accumulation and logging
        self.batch_accum_counter = 0
        self.total_loss = torch.tensor(0.0)
        self.total_tokens_processed = 0
        self.start_time = time.time()

        # Save any hyperparameters
        self.save_hyperparameters()

    def forward(self, x, y=None):
        return self.model(x, y)

    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(self.weight_decay, self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.get_scheduler())
        return [optimizer], [scheduler]

    def get_scheduler(self):
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return self.max_coefficient * (current_step + 1) / (self.warmup_steps)
            elif current_step > self.batched_steps:
                return self.min_coefficient
            else:
                decay_ratio = (current_step - self.warmup_steps) / (self.batched_steps - self.warmup_steps)
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                return self.min_coefficient + coeff * (self.max_coefficient - self.min_coefficient)
        return lr_lambda

    def training_step(self, batch, batch_idx):
        # See https://lightning.ai/docs/pytorch/stable/common/optimization.html for gradient accumulation and manual backward
        self.model.train()

        x = batch[:-1].view(self.B, self.T)
        y = batch[1:].view(self.B, self.T)

        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
            logits, loss = self(x, y)

        loss = loss / self.grad_accum_steps
        self.total_loss += loss.detach().cpu()
        self.batch_accum_counter += 1

        # Manual backward
        self.manual_backward(loss) 

        if self.batch_accum_counter == self.grad_accum_steps:
            # Gradient clipping. We don't need to rescale gradients since we are using bfloat16 and not fp16.
            # norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0, error_if_nonfinite=True)

            # Optimizer step
            logging.warning(f"Enzo: changed optimizer step")
            self.optimizers().optimizer.step()
            self.optimizers().optimizer.zero_grad()
            
            # Scheduler step
            self.lr_schedulers().step()

            # Logging
            lr = self.optimizers().param_groups[0]['lr']
            cuda_memory = torch.cuda.memory_allocated() if "cuda" in device else 0
            cpu_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)  # in GB

            current_time = time.time()
            dt = current_time - self.start_time
            tokens_per_second = (self.B * self.T * self.grad_accum_steps) / dt
            self.total_tokens_processed = self.B * self.T * batch_idx
            eta = (10**10 - self.total_tokens_processed) / tokens_per_second if tokens_per_second > 0 else float('inf')

            # Log metrics
            self.log('training_loss', self.total_loss, on_step=True, on_epoch=False, prog_bar=True)
            self.log('train/lr', lr, on_step=True, on_epoch=False)
            self.log('train/tokens_per_second', tokens_per_second, on_step=True, on_epoch=False)
            self.log('train/total_tokens_processed', self.total_tokens_processed, on_step=True, on_epoch=False)
            self.log('train/ETA_to_10B_tokens', eta, on_step=True, on_epoch=False)
            self.log('train/CUDA_memory_MB', cuda_memory / (1024 ** 2), on_step=True, on_epoch=False)
            self.log('train/CPU_memory_GB', cpu_memory, on_step=True, on_epoch=False)
            # self.log("train/norm", norm, on_step=True, on_epoch=False)
            # Reset counters
            self.total_loss = torch.tensor(0.0)
            self.batch_accum_counter = 0
            self.start_time = time.time()

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        x = batch[:-1].view(self.B, self.T)
        y = batch[1:].view(self.B, self.T)
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
            with torch.no_grad():
                logits, loss = self(x, y)
        self.log('val/loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss


# Function to get the most likely row in HellaSwag evaluation
def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# Hyperparameters
B = 16
total_batch_size = 524288

T = 1024
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)

steps_save = 10000
max_steps = 19500 * grad_accum_steps

# Initialize model
config = GPTConfig(vocab_size=50304, block_size=1024)
model = GPTLightningModule(config, total_batch_size, B, T, grad_accum_steps, max_steps, disable_compilation=disable_compilation)
# model = GPTLightningModule.load_from_checkpoint("/home/enzo/Documents/gpt2/saved_files/gpt2/5ykp6a5e/checkpoints/epoch=0-step=9000.ckpt",)

# DataLoader
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.worker_index = worker_info.id
    dataset.num_workers = worker_info.num_workers

train_dataloader = DataLoader(GPTDataset(B, T, "train"),
                              num_workers=1,
                              batch_size=None,
                              shuffle=False,
                              worker_init_fn=worker_init_fn,)
# val_dataloader = DataLoader(GPTDataset(B, T, "val"),
#                             num_workers=1,
#                             batch_size=None,
#                             shuffle=False,
#                             pin_memory=False,)

# Callbacks
checkpoint_callback = ModelCheckpoint(every_n_train_steps=steps_save)

# Trainer
trainer = pl.Trainer(
    max_steps=max_steps,
    log_every_n_steps=50,
    logger=wandb_logger,
    devices=1,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    # precision='bf16-mixed',
)

# Start training
trainer.fit(model,
            train_dataloader,)
