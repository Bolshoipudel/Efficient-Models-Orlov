import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bias import attention_stats
from model import build

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT / "results" / "runs"
HEADER_BYTES = 1024
BATCH_TOKENS = 16384
HEAD_CHUNK = 2048
VAL_TOKENS = 2**21
CURVE_TOKENS = 2**19
BIAS_TOKENS = 2**18
LOG_EVERY = 50
EVAL_EVERY = 500
TAIL_STEPS = 200


def load_tokens(prefix):
    files = sorted(DATA.glob(f"{prefix}_*.bin"))
    if not files:
        raise FileNotFoundError(f"нет файлов {prefix}_*.bin в {DATA}, сначала запустите data.py")
    return np.concatenate([np.fromfile(f, dtype=np.uint16, offset=HEADER_BYTES) for f in files])


def get_batch(data, rng, batch, seq_len):
    starts = rng.integers(0, len(data) - seq_len - 1, size=batch)
    chunk = torch.from_numpy(data[starts[:, None] + np.arange(seq_len + 1)].astype(np.int64)).cuda()
    return chunk[:, :-1], chunk[:, 1:]


def lr_at(step, steps, warmup, lr):
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))


def chunked_loss(model, h, targets, backward=False):
    h, targets = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
    total = torch.zeros((), device=h.device)
    for i in range(0, len(targets), HEAD_CHUNK):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.head(h[i : i + HEAD_CHUNK])
            loss = F.cross_entropy(logits, targets[i : i + HEAD_CHUNK], reduction="sum") / len(targets)
        if backward:
            loss.backward()
        total += loss.detach()
    return total


def train_step(model, x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.hidden(x)
    # логиты 16K x 50K целиком не помещаются в 12 ГБ, поэтому голова и loss считаются чанками:
    # градиент по h копится в leaf.grad и затем одним вызовом уходит в тело сети
    leaf = h.detach().requires_grad_()
    loss = chunked_loss(model, leaf, y, backward=True)
    h.backward(leaf.grad)
    return loss


@torch.no_grad()
def evaluate(model, val, seq_len, batch, n_tokens):
    model.eval()
    x = val[:n_tokens].view(-1, seq_len)
    y = val[1 : n_tokens + 1].view(-1, seq_len)
    total = 0.0
    for i in range(0, len(x), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.hidden(x[i : i + batch])
        total += chunked_loss(model, h, y[i : i + batch]).item() * len(h)
    model.train()
    return total / len(x)


def train(args):
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = build(args.size, xsa=args.xsa, n_sinks=args.sinks, no_self=args.no_self).cuda()
    total, body = model.n_params()
    groups = [
        {"params": [p for p in model.parameters() if p.dim() >= 2], "weight_decay": 0.1},
        {"params": [p for p in model.parameters() if p.dim() < 2], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95), fused=True)

    data = load_tokens("fineweb_train")
    val = torch.from_numpy(load_tokens("fineweb_val")[: VAL_TOKENS + 1].astype(np.int64)).cuda()
    rng = np.random.default_rng(args.seed)
    batch = BATCH_TOKENS // args.seq_len
    print(f"{args.name}: {body / 1e6:.1f}M params (+{(total - body) / 1e6:.1f}M emb), batch {batch} x {args.seq_len}, {args.steps} steps", flush=True)

    log_f = open(out / "log.csv", "w", newline="")
    eval_f = open(out / "eval.csv", "w", newline="")
    log_w, eval_w = csv.writer(log_f), csv.writer(eval_f)
    log_w.writerow(["step", "train_loss", "lr", "tokens_per_s"])
    eval_w.writerow(["step", "val_loss"])

    losses, tail, speeds = [], [], []
    diverged = False
    train_seconds = 0.0
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        lr = lr_at(step - 1, args.steps, args.warmup, args.lr)
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = get_batch(data, rng, batch, args.seq_len)
        loss = train_step(model, x, y)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss)

        if step % LOG_EVERY == 0 or step == args.steps:
            values = torch.stack(losses).float().tolist()
            dt = time.perf_counter() - t0
            train_seconds += dt
            speeds.append(len(values) * batch * args.seq_len / dt)
            tail = (tail + values)[-TAIL_STEPS:]
            log_w.writerow([step, f"{np.mean(values):.5f}", f"{lr:.3e}", f"{speeds[-1]:.0f}"])
            log_f.flush()
            losses = []
            if not all(math.isfinite(v) for v in values):
                diverged = True
                print(f"  step {step}: loss не конечен, остановка", flush=True)
                break
            if step % EVAL_EVERY == 0 or step == args.steps:
                val_loss = evaluate(model, val, args.seq_len, batch, CURVE_TOKENS)
                eval_w.writerow([step, f"{val_loss:.5f}"])
                eval_f.flush()
                print(f"  step {step:6d}  train {np.mean(values):.4f}  val {val_loss:.4f}  {speeds[-1] / 1e3:.1f}K tok/s", flush=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
    log_f.close()
    eval_f.close()
    peak = torch.cuda.max_memory_allocated()

    summary = {
        "name": args.name,
        "size": args.size,
        "xsa": args.xsa,
        "sinks": args.sinks,
        "no_self": args.no_self,
        "lr": args.lr,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "steps": args.steps,
        "warmup": args.warmup,
        "batch": batch,
        "params_total": total,
        "params_body": body,
        "diverged": diverged,
        "train_loss": float("nan") if diverged else float(np.mean(tail)),
        "val_loss": float("nan") if diverged else evaluate(model, val, args.seq_len, batch, VAL_TOKENS),
        "tokens_per_s": float(np.median(speeds)),
        "peak_memory_bytes": peak,
        "train_seconds": train_seconds,
        "bias": [] if diverged else attention_stats(model, val[:BIAS_TOKENS], args.seq_len),
    }
    if args.save:
        torch.save(model.state_dict(), out / "ckpt.pt")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"  итог: train {summary['train_loss']:.4f}  val {summary['val_loss']:.4f}  {summary['tokens_per_s'] / 1e3:.1f}K tok/s  peak {peak / 2**30:.2f} GiB  {train_seconds / 60:.1f} мин", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--size", default="S")
    parser.add_argument("--xsa", action="store_true")
    parser.add_argument("--sinks", type=int, default=0)
    parser.add_argument("--no-self", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action="store_true")
    train(parser.parse_args())
