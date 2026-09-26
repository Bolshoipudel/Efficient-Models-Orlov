import argparse
import csv
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np
import pynvml
import torch

from models import build_model

BASE_SIZES = [32, 64, 128, 224, 256, 384, 512]
BASE_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
RESULTS = Path(__file__).resolve().parent / "results"
FIELDS = ["image_size", "batch", "latency_s", "memory_bytes", "energy_j", "is_validation"]
OOM_FIELDS = ["image_size", "batch", "limit_bytes", "memory_bytes", "is_validation"]


def sample_grid(seed):
    rng = np.random.default_rng(seed)
    size_pool = [s for s in range(32, 513, 16) if s not in BASE_SIZES]
    batch_pool = [b for b in range(1, 257) if b & (b - 1)]
    sizes = sorted(int(s) for s in rng.choice(size_pool, 4, replace=False))
    batches = sorted(int(b) for b in rng.choice(batch_pool, 3, replace=False))
    return sizes, batches


def setup():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(1.0)


def forward(model, x):
    with torch.inference_mode():
        model(x)


def measure_config(model, handle, s, b, warmup, min_time):
    x = torch.randn(b, 3, s, s, device="cuda")
    for _ in range(warmup):
        forward(model, x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    forward(model, x)
    torch.cuda.synchronize()
    memory = torch.cuda.max_memory_allocated()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    t0 = time.perf_counter()
    while len(times) < 5 or (time.perf_counter() - t0 < min_time and len(times) < 200):
        start.record()
        forward(model, x)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    latency = float(np.median(times))

    n = max(10, int(np.ceil(min_time / latency)))
    torch.cuda.synchronize()
    e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    for _ in range(n):
        forward(model, x)
    torch.cuda.synchronize()
    e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    energy = (e1 - e0) / 1e3 / n

    return latency, memory, energy


def peak_under_limit(model, s, b):
    x = torch.randn(b, 3, s, s, device="cuda")
    forward(model, x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    forward(model, x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated()


def run_oom(model, sizes, batches, rand_sizes, rand_batches, limit_gib):
    limit = int(limit_gib * 2**30)
    torch.cuda.set_per_process_memory_fraction(limit / torch.cuda.get_device_properties(0).total_memory)
    with open(RESULTS / "oom.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OOM_FIELDS)
        writer.writeheader()
        for s in sizes:
            for b in batches:
                row = {"image_size": s, "batch": b, "limit_bytes": limit, "is_validation": s in rand_sizes or b in rand_batches}
                try:
                    row["memory_bytes"] = peak_under_limit(model, s, b)
                    print(f"S={s:4d} B={b:4d}  {row['memory_bytes'] / 2**20:9.1f} MiB")
                except torch.cuda.OutOfMemoryError:
                    row["memory_bytes"] = "OOM"
                    print(f"S={s:4d} B={b:4d}  OOM")
                gc.collect()
                torch.cuda.empty_cache()
                writer.writerow(row)
                f.flush()


def save_env(handle, seed, sizes, batches):
    props = torch.cuda.get_device_properties(0)
    env = {
        "gpu": props.name,
        "gpu_memory_bytes": props.total_memory,
        "driver": pynvml.nvmlSystemGetDriverVersion(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "seed": seed,
        "image_sizes": sizes,
        "batches": batches,
    }
    (RESULTS / "env.json").write_text(json.dumps(env, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--min-time", type=float, default=1.0)
    parser.add_argument("--oom-limit-gib", type=float, default=None)
    args = parser.parse_args()

    setup()
    model = build_model().cuda()
    rand_sizes, rand_batches = sample_grid(args.seed)
    sizes = sorted(BASE_SIZES + rand_sizes)
    batches = sorted(BASE_BATCHES + rand_batches)
    RESULTS.mkdir(exist_ok=True)

    if args.oom_limit_gib is not None:
        run_oom(model, sizes, batches, rand_sizes, rand_batches, args.oom_limit_gib)
        return

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
    save_env(handle, args.seed, sizes, batches)

    with open(RESULTS / "measurements.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for s in sizes:
            for b in batches:
                row = {"image_size": s, "batch": b, "is_validation": s in rand_sizes or b in rand_batches}
                try:
                    latency, memory, energy = measure_config(model, handle, s, b, args.warmup, args.min_time)
                    row.update(latency_s=latency, memory_bytes=memory, energy_j=energy)
                    print(f"S={s:4d} B={b:4d}  {latency * 1e3:9.3f} ms  {memory / 2**20:9.1f} MiB  {energy:8.4f} J")
                except torch.cuda.OutOfMemoryError:
                    row.update(latency_s="", memory_bytes="OOM", energy_j="")
                    print(f"S={s:4d} B={b:4d}  OOM")
                gc.collect()
                torch.cuda.empty_cache()
                writer.writerow(row)
                f.flush()

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
