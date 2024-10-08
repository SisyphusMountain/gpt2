from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import tiktoken; enc = tiktoken.get_encoding("gpt2")
from hellaswag import render_example, iterate_examples
import itertools
import math
import time
import os
import logging
import argparse
import psutil
from torch.utils.data import IterableDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
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

disable_compilation = False
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

    def __iter__(self):
        shard_iter = itertools.cycle(self.shards)
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
        self.config = config
        self.B = B
        self.T = T
        self.total_batch_size = total_batch_size
        self.grad_accum_steps = grad_accum_steps
        self.max_steps = max_steps

        self.lr = 6e-4
        self.weight_decay = 0.1
        # Scheduler parameters
        self.max_coefficient = 1.0
        self.min_coefficient = self.max_coefficient * 0.1
        self.warmup_steps = 715

        # Log any hyperparameters
        self.save_hyperparameters()

    def forward(self, x, y=None):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        x = batch[:-1].view(self.B, self.T)
        y = batch[1:].view(self.B, self.T)
        logits, loss = self(x, y)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        x = batch[:-1].view(self.B, self.T)
        y = batch[1:].view(self.B, self.T)
        logits, loss = self(x, y)
        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss
    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(self.weight_decay, self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.get_scheduler())
        return [optimizer], [scheduler]

    def get_scheduler(self):
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return self.max_coefficient * (current_step + 1) / (self.warmup_steps)
            elif current_step > self.max_steps:
                return self.min_coefficient
            else:
                decay_ratio = (current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                return self.min_coefficient + coeff * (self.max_coefficient - self.min_coefficient)
        return lr_lambda

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

from collections import deque

# Custom Callback to compute and log tokens per second and ETA to 10B tokens using a sliding window
class TokenSpeedLogger(pl.Callback):
    def __init__(self, total_tokens_target=10**10, window_size=100):
        super().__init__()
        self.total_tokens_target = total_tokens_target
        self.window_size = window_size
        self.token_timestamps = deque()
        self.total_tokens_processed = 0
        self.start_time = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self.token_timestamps.append((self.total_tokens_processed, self.start_time))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Number of tokens in this batch
        tokens_in_batch = pl_module.B * pl_module.T
        self.total_tokens_processed += tokens_in_batch

        current_time = time.time()
        self.token_timestamps.append((self.total_tokens_processed, current_time))

        # Keep only the last 'window_size' entries
        if len(self.token_timestamps) > self.window_size:
            self.token_timestamps.popleft()

        # Compute tokens per second over the sliding window
        delta_tokens = self.total_tokens_processed - self.token_timestamps[0][0]
        delta_time = current_time - self.token_timestamps[0][1]
        tokens_per_second = delta_tokens / delta_time if delta_time > 0 else 0.0

        # Estimated time to reach total_tokens_target
        tokens_remaining = self.total_tokens_target - self.total_tokens_processed
        eta = tokens_remaining / tokens_per_second if tokens_per_second > 0 else float('inf')

        # Log metrics
        metrics = {
            'tokens_per_second': tokens_per_second,
            'estimated_time_to_10B_tokens': eta,
            'total_tokens_processed': self.total_tokens_processed,
        }

        # Log to wandb using the logger
        trainer.logger.log_metrics(metrics, step=trainer.global_step)

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
grad_accum_steps = total_batch_size // (B * T)

steps_save = 1000
max_steps = 19500 * grad_accum_steps

# Initialize model
config = GPTConfig(vocab_size=50304, block_size=1024)
model = GPTLightningModule(config, total_batch_size, B, T, grad_accum_steps, max_steps, disable_compilation=disable_compilation)

# DataLoader
train_dataloader = DataLoader(GPTDataset(B, T, "train"),
                              batch_size=None,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True,
                              pin_memory_device=device)
val_dataloader = DataLoader(GPTDataset(B, T, "val"),
                            batch_size=None,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=False,)

# Callbacks
checkpoint_callback = ModelCheckpoint(every_n_train_steps=steps_save)
token_speed_logger = TokenSpeedLogger(total_tokens_target=10**10)
# Trainer
trainer = pl.Trainer(
    max_steps=max_steps,
    limit_val_batches=20,
    val_check_interval=1024,
    accumulate_grad_batches=grad_accum_steps,
    gradient_clip_val=1.0,
    gradient_clip_algorithm='norm',
    logger=wandb_logger,
    callbacks=[checkpoint_callback, token_speed_logger],
    devices=1,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    precision='bf16',
)

# Start training
trainer.fit(model, train_dataloader, val_dataloader)
