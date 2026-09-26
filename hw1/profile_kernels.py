import csv
from collections import defaultdict

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from measure import RESULTS, forward, setup
from models import build_model

CONFIGS = [(32, 1), (128, 2), (128, 4), (256, 32), (256, 64), (256, 128), (512, 32), (512, 64), (512, 128)]


def kernels_of_one_pass(model, s, b, warmup=3):
    x = torch.randn(b, 3, s, s, device="cuda")
    for _ in range(warmup):
        forward(model, x)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        forward(model, x)
        torch.cuda.synchronize()
    stats = defaultdict(lambda: [0, 0.0])
    order = []
    for e in prof.events():
        if e.device_type != DeviceType.CUDA:
            continue
        if e.name not in stats:
            order.append(e.name)
        stats[e.name][0] += 1
        stats[e.name][1] += e.device_time / 1e3
    return [(name, *stats[name]) for name in order]


def short(name, width=90):
    return name if len(name) <= width else name[: width - 3] + "..."


def main():
    setup()
    model = build_model().cuda()
    rows = []
    for s, b in CONFIGS:
        kernels = kernels_of_one_pass(model, s, b)
        total = sum(k[1] for k in kernels)
        busy = sum(k[2] for k in kernels)
        print(f"\nS={s} B={b}: {total} kernel launches, GPU busy {busy:.3f} ms")
        for name, count, ms in kernels:
            print(f"  {count:2d} x {ms:8.3f} ms  {short(name)}")
            rows.append({"image_size": s, "batch": b, "kernel": name, "launches": count, "gpu_time_ms": ms})
    with open(RESULTS / "kernels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_size", "batch", "kernel", "launches", "gpu_time_ms"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
