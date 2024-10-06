from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import tiktoken
import time
import math
import os
from datasets import load_dataset
"""Ideas:
- Only warmup attention"""
disable_compilation = True
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

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
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
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

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {"train", "val"}
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        # with open('input.txt', 'r') as f:
        #     text = f.read()
        # enc = tiktoken.get_encoding("gpt2")
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)
        # print(f"loaded {len(tokens)} tokens")
        # print(f"number of epochs = {len(tokens)//(B*T)}")

        
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B*T + 1]
        y = buf[1:].view(B, T)
        x = buf[:-1].view(B, T)
        self.current_position += B*T
        if self.current_position + (B*T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y


# model = GPT(GPTConfig(vocab_size=50304))
model = GPT(GPTConfig(vocab_size=50304, block_size=1024))

model.to(device)
model.train()
#print(f"cuda memory used after creating model: {torch.cuda.memory_allocated()}")
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4)
torch.set_float32_matmul_precision("high")

# total_batch_size = 512000 # 25*20*1024, where 20 is the batch size for each step, 1024 is the sequence length for each batch, and 25 will be the number of accumulation steps
total_batch_size = 524288

B = 16
T = 1024
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size {total_batch_size}")
print(f"=> calculated gradient accumulation steps {grad_accum_steps}")
train_loader = DataLoaderLite(B=B, T=T, split="train")


max_coefficient = 1
min_coefficient = max_coefficient * 0.1
warmup_steps = 715 # Karpathy says it can be made smaller
max_steps = 19500

def get_scheduler(max_coefficient, min_coefficient, warmup_steps, max_steps):
    """Karpathy says to use the LambdaLR scheduler, because it is simple, and 
    other ways of scheduling via PyTorch implementations are more obscure."""
    def get_lr(step):
        """There are three stages for learning rate scheduling:
        1) warmup, before warmup steps
        2) between warmup steps and max steps
        3) after max steps"""
        if step < warmup_steps:
            return max_coefficient * (step+1)/(warmup_steps)
        elif step > max_steps:
            return min_coefficient
        else:
            decay_ratio = (step - warmup_steps)/(max_steps - warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_coefficient + coeff * (max_coefficient - min_coefficient)
    return get_lr

# I should compile the whole training step: https://discuss.pytorch.org/t/torch-compile-what-is-the-best-scope-of-compilation/185442/6
# However, when I add the scheduler, I run into problems. It is easier to separate them
@torch.compile(fullgraph=False, disable=disable_compilation)
def optim_step():
    optimizer.step()
    scheduler.step()

@torch.compile(mode="max-autotune", disable=disable_compilation)
def training_step(x, y, grad_accum_steps):
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss = loss / grad_accum_steps # Make sure the MSE is computed as a mean over the whole batch
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters=model.parameters(),
                                          max_norm=1.0,
                                          error_if_nonfinite=True,
                                          ) # WARNING: can error_if_nonfinite cause an error or slow down training?

    return loss, norm

 
get_lr = get_scheduler(max_coefficient,
                          min_coefficient,
                          warmup_steps,
                          max_steps)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

mean = 0
for step in range(max_steps):
    t0 = time.perf_counter()
    total_loss = torch.tensor(0.0)
    optimizer.zero_grad()
    for mini_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        

        loss, norm = training_step(x, y, grad_accum_steps)
        total_loss += loss.detach().cpu()
    loss = total_loss.item()
    optim_step()
    torch.cuda.synchronize() # Necessary to measure time accurately. Otherwise, the time will be taken asynchronously before the end of the training step on GPU.
    t1 = time.perf_counter()
    dt = (t1 - t0)
    tokens_per_second = (B*T*grad_accum_steps)/dt
    print(f"tokens per second = {tokens_per_second:.0f} | loss: {loss:.6f} | gradient norm {norm:.4f} | dt {dt*1000:.6f} ms")

