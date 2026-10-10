import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

VOCAB_SIZE = 50304
SIZES = {
    "S": dict(n_layer=6, d_model=384, n_head=6, head_dim=64),
    "M": dict(n_layer=8, d_model=512, n_head=8, head_dim=64),
    "L": dict(n_layer=12, d_model=768, n_head=12, head_dim=64),
}


@dataclass
class GPTConfig:
    n_layer: int = 6
    d_model: int = 384
    n_head: int = 6
    head_dim: int = 64
    xsa: bool = False
    n_sinks: int = 0
    vocab_size: int = VOCAB_SIZE
    no_self: bool = False


def rope_tables(seq_len, head_dim, device, base=10000.0):
    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    angles = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos + x2 * sin, x2 * cos - x1 * sin], dim=-1)


def strict_causal(n, device):
    # j < i; первый токен оставляем видеть себя, иначе его строка softmax пуста
    mask = torch.ones(n, n, dtype=torch.bool, device=device).tril(-1)
    mask[0, 0] = True
    return mask


def exclude_self(y, v):
    vn = F.normalize(v, dim=-1)
    return y - (y * vn).sum(dim=-1, keepdim=True) * vn


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.head_dim, self.xsa = cfg.n_head, cfg.head_dim, cfg.xsa
        self.no_self = cfg.no_self
        inner = cfg.n_head * cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * inner, bias=False)
        self.proj = nn.Linear(inner, cfg.d_model, bias=False)

    def heads(self, x, rope):
        B, T, _ = x.shape
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in self.qkv(x).chunk(3, dim=-1))
        return apply_rope(q, *rope), apply_rope(k, *rope), v

    def merge(self, y):
        B, _, T, _ = y.shape
        return self.proj(y.transpose(1, 2).reshape(B, T, -1))

    def forward(self, x, rope):
        q, k, v = self.heads(x, rope)
        if self.no_self:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=strict_causal(x.shape[1], x.device))
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if self.xsa:
            y = exclude_self(y, v)
        return self.merge(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
        self.mlp = MLP(cfg)

    def forward(self, x, rope):
        x = x + self.attn(self.ln1(x), rope)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.ln_emb = nn.LayerNorm(cfg.d_model, bias=False)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model, bias=False)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))
        # sink-токены создаются последними, чтобы остальные веса совпадали с моделью без них
        self.sinks = nn.Parameter(torch.randn(1, cfg.n_sinks, cfg.d_model) * 0.02) if cfg.n_sinks else None

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def embed(self, idx):
        x = self.wte(idx)
        if self.sinks is not None:
            x = torch.cat([self.sinks.expand(x.shape[0], -1, -1).to(x.dtype), x], dim=1)
        return self.ln_emb(x)

    def hidden(self, idx):
        x = self.embed(idx)
        rope = rope_tables(x.shape[1], self.cfg.head_dim, x.device)
        for block in self.blocks:
            x = block(x, rope)
        return x[:, self.cfg.n_sinks:]

    def head(self, h):
        return self.lm_head(self.ln_f(h))

    def forward(self, idx):
        return self.head(self.hidden(idx))

    def n_params(self):
        total = sum(p.numel() for p in self.parameters())
        return total, total - self.wte.weight.numel()


def build(size, **overrides):
    return GPT(GPTConfig(**{**SIZES[size], **overrides}))


if __name__ == "__main__":
    for size in SIZES:
        total, body = build(size).n_params()
        print(f"{size}: всего {total / 1e6:.1f}M, без эмбеддингов {body / 1e6:.1f}M")
    x = torch.randn(2, 4, 16, 64)
    v = torch.randn(2, 4, 16, 64)
    z = exclude_self(x, v)
    print(f"max |cos(z, v)| после проекции: {F.cosine_similarity(z, v, dim=-1).abs().max():.2e}")
