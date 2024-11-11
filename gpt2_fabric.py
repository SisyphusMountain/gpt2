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
import random
import psutil
from torch.utils.data import IterableDataset, DataLoader
torch.set_float32_matmul_precision("high")

import wandb
run = wandb.init(project="gpt2")

parser = argparse.ArgumentParser()
parser.add_argument('--log', default='info', choices=['debug', 'info', 'warning', 'error', 'critical'],
                    help="Set the logging level (default: info)")
args = parser.parse_args()
log_level = getattr(logging, args.log.upper(), logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

device = "cuda:0" if torch.cuda.is_available() else "cpu"

disable_compilation=False
if disable_compilation:
    logging.warning("Disabling compilation")
else:
    logging.info("Enabling compilation")

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

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
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

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
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
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
        self.transformer.wte.weight = self.lm_head.weight  # Weight sharing
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
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
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
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
grad_accum_steps = total_batch_size // (B * T)

# Using the adjusted DataLoaderLite
train_dataloader = GPTDataset(B=B, T=T, split="train")

model = GPT(GPTConfig(vocab_size=50304, block_size=1024))

model.to(device)
model.train()
logging.info(f"used cuda memory after creating model: {torch.cuda.memory_allocated()}")
optimizer = model.configure_optimizers(weight_decay=0.1,
                                       learning_rate=6e-4,)

# Compile the model, training_step, and optim_step functions
model = torch.compile(model, mode="max-autotune", disable=disable_compilation)

max_coefficient = 1.0
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

get_lr = get_scheduler(max_coefficient,
                       min_coefficient,
                       warmup_steps,
                       max_steps)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

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

logging.debug("""Please make sure that we actually want to do a certain number of steps,
                rather than a certain number of epochs.""")
steps_save = 1000
step_counter = 0
batch_accum_counter = 0
total_processed_tokens = 0
temp_step_counter = 0 
total_loss = torch.tensor(0.0)
t0 = time.time()
t1 = time.time()

# Initialize optimizer zero_grad before starting the loop
optimizer.zero_grad()

# Define the training_step and optim_step functions and compile them
#@torch.compile(mode="max-autotune", disable=disable_compilation)
def training_step(x, y):
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss = loss / grad_accum_steps
    loss.backward()
    return loss.detach().cpu()

#@torch.compile(mode="max-autotune", disable=disable_compilation)
def optim_step():
    # Perform gradient clipping
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    return norm

for buf in train_dataloader:
    x = buf[:-1].view(B, T)
    y = buf[1:].view(B, T)
    x = x.to(device)
    y = y.to(device)
    logging.debug(f"In the training loop, x shape {x.shape}, y shape {y.shape}")
    loss = training_step(x, y)
    total_loss += loss
    batch_accum_counter += 1

    if batch_accum_counter == grad_accum_steps:
        batch_accum_counter = 0
        norm = optim_step()
        step_counter += 1
        temp_step_counter += 1

        cuda_memory = torch.cuda.memory_allocated() if "cuda" in device else 0
        cpu_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)  # in GB

        last_lr = scheduler.get_last_lr()[0]
        t0 = t1
        t1 = time.time()
        dt = (t1 - t0)
        tokens_per_second = B * T * grad_accum_steps / dt
        total_processed_tokens += B * T * grad_accum_steps
        logging.info(f"tokens per second = {tokens_per_second:.0f} | loss: {total_loss.item():.6f} | "
                        f"gradient norm {norm:.4f} | dt {dt*1000:.6f} ms | "
                        f"CUDA memory: {cuda_memory / (1024 ** 2):.2f} MB | "
                        f"CPU memory: {cpu_memory:.2f} GB"
                        f"step {step_counter}")
        
        run.log({"tokens per second": tokens_per_second,
                                    "loss": total_loss.item(),
                                    "gradient norm": norm,
                                    "dt": dt,
                                    "CUDA memory": cuda_memory / (1024 ** 2),
                                    "CPU memory": cpu_memory,
                                    "learning rate": last_lr,
                                    "total_processed_tokens": total_processed_tokens,
                                    "step": step_counter,
                                    "ETA to 10 billion tokens": (10**10 - total_processed_tokens) / tokens_per_second},
                                    step=step_counter)
        total_loss = torch.tensor(0.0)

        if temp_step_counter == steps_save:
            logging.info("Saving model")
            torch.save(model.state_dict(), f"model_{step_counter}.pt")
            logging.info("Model saved")
            model.eval()

            # Validation
            val_dataloader = GPTDataset(B=B, T=T, split="val")
            loss_val = torch.tensor(0.0)
            loss_val_steps = 20
            for buf_val in range(loss_val_steps):
                buf_val = next(val_dataloader)
                x_val = buf_val[:-1].view(B, T)
                y_val = buf_val[1:].view(B, T)
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, loss = model(x_val, y_val)
                loss_val += loss.cpu()
            logging.info(f"Validation loss: {loss_val.item() / loss_val_steps}")
            run.log({"Validation loss": loss_val.item() / loss_val_steps}, step=step_counter)

            # Sample Generation
            num_return_sequences = 5
            max_length = 32
            initial_prompt = "Hello, I am a language model,"

            tokens = enc.encode(initial_prompt)
            tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            tokens = tokens.repeat(num_return_sequences, 1)
            xgen = tokens.clone()
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42)

            while xgen.size(1) < max_length:
                with torch.no_grad():
                    logits, _ = model(xgen)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=50, dim=-1)
                next_tokens = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
                next_tokens = topk_indices.gather(dim=-1, index=next_tokens)
                xgen = torch.cat([xgen, next_tokens], dim=1)

            result_strings = []
            for i in range(num_return_sequences):
                tokens = xgen[i].cpu().numpy().tolist()
                decoded = enc.decode(tokens)
                result_strings.append(decoded)
                logging.info(f"Sample {i+1}: {decoded}")

            table = wandb.Table(columns=["Sample"])
            for sample in result_strings:
                table.add_data(sample)
            run.log({"generated_samples": table}, step=step_counter)

            # Evaluating on Hellaswag
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                _, tokens, mask, label = render_example(example)
                tokens = tokens.to(device)
                mask = mask.to(device)

                with torch.no_grad():
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        logits, _ = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            
            accuracy_norm = num_correct_norm / num_total
            logging.info(f"Accuracy on Hellaswag: {accuracy_norm}")
            run.log({"Accuracy on Hellaswag": accuracy_norm}, step=step_counter)

            model.train()
            temp_step_counter = 0

    if step_counter >= max_steps:
        break
