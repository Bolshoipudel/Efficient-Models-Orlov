import csv
import gc
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from model import Block, GPTConfig, exclude_self, rope_tables
from xsa_fast import FastXSA, exclude_self_fast

RESULTS = Path(__file__).resolve().parent / "results"
BATCH = 16
HEAD_DIM = 64
SEQ_LENS = [256, 512, 1024, 2048, 4096, 8192]
D_MODELS = [256, 384, 512, 768, 1024, 1536]
FIXED_D = 512
FIXED_T = 1024
ROUNDS = 3
VARIANTS = ["baseline", "xsa", "xsa_fast"]
FIELDS = ["sweep", "d_model", "seq_len", "batch", "mode", "variant", "time_ms", "memory_mb"]
KERNEL_FIELDS = ["variant", "kernel", "launches", "gpu_time_ms"]


class Bench:
    def __init__(self, d_model, seq_len):
        cfg = GPTConfig(d_model=d_model, n_head=d_model // HEAD_DIM, head_dim=HEAD_DIM)
        self.block = Block(cfg).cuda()
        self.paper = self.block.attn
        self.fast = FastXSA(cfg).cuda()
        self.fast.qkv, self.fast.proj = self.paper.qkv, self.paper.proj
        self.x = torch.randn(BATCH, seq_len, d_model, device="cuda", requires_grad=True)
        self.grad = torch.randn(BATCH, seq_len, d_model, device="cuda")
        self.rope = rope_tables(seq_len, HEAD_DIM, "cuda")

    def use(self, variant):
        self.block.attn = self.fast if variant == "xsa_fast" else self.paper
        self.paper.xsa = variant == "xsa"

    def step(self, backward):
        if not backward:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                self.block(self.x, self.rope)
            return
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.block(self.x, self.rope)
        out.backward(self.grad)
        self.block.zero_grad(set_to_none=True)
        self.x.grad = None

    def peak_memory(self, backward):
        self.step(backward)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.step(backward)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    def timings(self, backward, min_time=0.5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        t0 = time.perf_counter()
        while len(times) < 5 or (time.perf_counter() - t0 < min_time and len(times) < 100):
            start.record()
            self.step(backward)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
        return times

    def kernels(self):
        for _ in range(3):
            self.step(True)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            self.step(True)
            torch.cuda.synchronize()
        stats = defaultdict(lambda: [0, 0.0])
        for e in prof.events():
            if e.device_type == DeviceType.CUDA:
                stats[e.name][0] += 1
                stats[e.name][1] += e.device_time / 1e3
        return stats


def measure(d_model, seq_len):
    bench = Bench(d_model, seq_len)
    rows = []
    for mode, backward in (("fwd", False), ("fwd+bwd", True)):
        memory, times = {}, defaultdict(list)
        for variant in VARIANTS:
            bench.use(variant)
            memory[variant] = bench.peak_memory(backward)
        # чередуем варианты, чтобы дрейф частоты и температуры GPU не попал в разницу
        for _ in range(ROUNDS):
            for variant in VARIANTS:
                bench.use(variant)
                times[variant] += bench.timings(backward)
        for variant in VARIANTS:
            rows.append({"d_model": d_model, "seq_len": seq_len, "batch": BATCH, "mode": mode, "variant": variant,
                         "time_ms": float(np.median(times[variant])), "memory_mb": memory[variant] / 2**20})
    return rows


def residual(fn, d_model=FIXED_D, seq_len=FIXED_T):
    y, v = (torch.randn(BATCH, d_model // HEAD_DIM, seq_len, HEAD_DIM, device="cuda") for _ in range(2))
    y = y + v
    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = fn(y.bfloat16(), v.bfloat16()).bfloat16()
    return F.cosine_similarity(z.float(), v, dim=-1).abs().mean().item()


def main():
    RESULTS.mkdir(exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    grid = [("seq_len", FIXED_D, t) for t in SEQ_LENS] + [("d_model", d, FIXED_T) for d in D_MODELS]
    with open(RESULTS / "overhead.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for sweep, d_model, seq_len in grid:
            try:
                rows = measure(d_model, seq_len)
            except torch.cuda.OutOfMemoryError:
                print(f"d={d_model:5d} T={seq_len:5d}  OOM")
                gc.collect()
                torch.cuda.empty_cache()
                continue
            for row in rows:
                writer.writerow({"sweep": sweep, **row})
            f.flush()
            for mode in ("fwd", "fwd+bwd"):
                cell = {r["variant"]: r for r in rows if r["mode"] == mode}
                base = cell["baseline"]
                over = "  ".join(f"{v}: time {cell[v]['time_ms'] / base['time_ms'] - 1:+.1%} mem {cell[v]['memory_mb'] / base['memory_mb'] - 1:+.1%}" for v in VARIANTS[1:])
                print(f"d={d_model:5d} T={seq_len:5d} {mode:8s} baseline {base['time_ms']:8.2f} ms {base['memory_mb']:6.0f} MB  {over}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()

    bench = Bench(FIXED_D, FIXED_T)
    with open(RESULTS / "overhead_kernels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KERNEL_FIELDS)
        writer.writeheader()
        for variant in VARIANTS:
            bench.use(variant)
            stats = bench.kernels()
            writer.writerows({"variant": variant, "kernel": name, "launches": n, "gpu_time_ms": ms} for name, (n, ms) in stats.items())
            print(f"{variant:9s} {sum(n for n, _ in stats.values()):4d} kernel launches, GPU busy {sum(ms for _, ms in stats.values()):6.2f} ms")
    for name, fn in (("xsa", exclude_self), ("xsa_fast", exclude_self_fast)):
        print(f"{name:9s} средний |cos(z, v)| в bf16: {residual(fn):.1e}")


if __name__ == "__main__":
    main()
