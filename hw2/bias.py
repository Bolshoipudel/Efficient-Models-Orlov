import torch
import torch.nn.functional as F

from model import exclude_self, rope_tables, strict_causal


@torch.no_grad()
def layer_stats(model, idx):
    cfg = model.cfg
    k0 = cfg.n_sinks
    x = model.embed(idx)
    n = x.shape[1]
    rope = rope_tables(n, cfg.head_dim, x.device)
    allowed = strict_causal(n, x.device) if cfg.no_self else torch.ones(n, n, dtype=torch.bool, device=x.device).tril()
    rows = []
    for block in model.blocks:
        q, k, v = block.attn.heads(block.ln1(x), rope)
        att = (q @ k.transpose(-2, -1)) / cfg.head_dim**0.5
        att = att.masked_fill(~allowed, float("-inf")).softmax(dim=-1)
        y = att @ v
        z = exclude_self(y, v) if cfg.xsa else y

        T = n - k0
        vn = F.normalize(v[:, :, k0:], dim=-1)
        row = {
            # среднее cos(v_i, v_j) по парам i != j: (|sum v_i|^2 - T) / (T (T - 1))
            "cos_vv": ((vn.sum(dim=2).pow(2).sum(dim=-1) - T) / (T * (T - 1))).mean().item(),
            "a_ii": att.diagonal(dim1=-2, dim2=-1)[..., k0 + 1:].mean().item(),
            "cos_yv": F.cosine_similarity(y, v, dim=-1)[..., k0:].mean().item(),
            "cos_zv": F.cosine_similarity(z, v, dim=-1)[..., k0:].mean().item(),
        }
        if k0:
            row["sink_mass"] = att[..., k0:, :k0].sum(dim=-1).mean().item()
        rows.append(row)

        x = x + block.attn.merge(z)
        x = x + block.mlp(block.ln2(x))
    return rows


def attention_stats(model, tokens, seq_len, batch_tokens=8192):
    was_training = model.training
    model.eval()
    seqs = tokens[: len(tokens) // seq_len * seq_len].view(-1, seq_len)
    batch = max(1, batch_tokens // seq_len)
    chunks = [layer_stats(model, seqs[i : i + batch]) for i in range(0, len(seqs), batch)]
    model.train(was_training)
    return [
        {key: sum(chunk[layer][key] for chunk in chunks) / len(chunks) for key in chunks[0][layer]}
        for layer in range(len(chunks[0]))
    ]
