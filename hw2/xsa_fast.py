import torch
import torch.nn.functional as F

from model import Attention, exclude_self

EPS = 1e-12


def dot(a, b):
    return (a * b).sum(dim=-1, keepdim=True, dtype=torch.float32 if a.dtype != torch.float64 else None)


class ExcludeSelf(torch.autograd.Function):
    # z = y - c v,  c = <y, v> / <v, v>: та же проекция, что в exclude_self, но без нормировки v,
    # без временных тензоров в fp32 и с выписанным вручную backward
    @staticmethod
    def forward(ctx, y, v):
        inv = dot(v, v).clamp_min(EPS).reciprocal()
        coef = (dot(y, v) * inv).to(y.dtype)
        z = torch.addcmul(y, v, coef, value=-1)
        ctx.save_for_backward(v, z, coef, inv)
        return z

    @staticmethod
    def backward(ctx, g):
        v, z, coef, inv = ctx.saved_tensors
        a = (dot(g, v) * inv).to(g.dtype)
        g_y = torch.addcmul(g, v, a, value=-1)
        g_v = torch.addcmul(-coef * g_y, z, a, value=-1)
        return g_y, g_v


def exclude_self_fast(y, v):
    return ExcludeSelf.apply(y, v)


class FastXSA(Attention):
    def forward(self, x, rope):
        q, k, v = self.heads(x, rope)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.merge(exclude_self_fast(y, v))


if __name__ == "__main__":
    y = torch.randn(2, 3, 5, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, 3, 5, 8, dtype=torch.float64, requires_grad=True)
    print("gradcheck:", torch.autograd.gradcheck(exclude_self_fast, (y, v)))
    print(f"max |fast - paper|: {(exclude_self_fast(y, v) - exclude_self(y, v)).abs().max():.2e}")
    g = torch.randn_like(y)
    fast = torch.autograd.grad(exclude_self_fast(y, v), (y, v), g)
    paper = torch.autograd.grad(exclude_self(y, v), (y, v), g)
    print(f"max |grad fast - grad paper|: {max((a - b).abs().max() for a, b in zip(fast, paper)):.2e}")
