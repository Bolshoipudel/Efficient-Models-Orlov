import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results" / "runs"
STEPS = 4000
LRS = [1e-3, 5e-4, 2e-3, 4e-3]
SEEDS = [1, 2]
SEQ_LENS = [256, 512, 2048, 4096]
SINKS = [1, 4]
# lr подбирается только для S, для M и L берётся то же отношение, что в Table 1 статьи (5 : 4 : 3)
LR_SCALE = {"S": 1.0, "M": 0.8, "L": 0.6}


def run_name(size, xsa, lr, seq_len, sinks, seed, no_self=False):
    return f"{size}_{'xsa' if xsa else 'sa'}_T{seq_len}_lr{lr:g}_k{sinks}_s{seed}" + "_noself" * no_self


def run(xsa, lr, size="S", seq_len=1024, sinks=0, seed=0, no_self=False, save=False):
    name = run_name(size, xsa, lr, seq_len, sinks, seed, no_self)
    if (RUNS / name / "summary.json").exists():
        return
    cmd = [sys.executable, str(ROOT / "train.py"), "--name", name, "--size", size, "--lr", str(lr)]
    cmd += ["--seq-len", str(seq_len), "--sinks", str(sinks), "--seed", str(seed), "--steps", str(STEPS)]
    cmd += ["--xsa"] * xsa + ["--no-self"] * no_self + ["--save"] * save
    subprocess.run(cmd, cwd=ROOT)


def pair(**kwargs):
    for xsa in (False, True):
        run(xsa, **kwargs)


def best_lr():
    scores = {}
    for lr in LRS:
        path = RUNS / run_name("S", False, lr, 1024, 0, 0) / "summary.json"
        if path.exists() and not (summary := json.loads(path.read_text()))["diverged"]:
            scores[lr] = summary["val_loss"]
    return min(scores, key=scores.get)


if __name__ == "__main__":
    for lr in LRS:
        pair(lr=lr, save=True)
    lr = best_lr()
    print(f"лучший lr для baseline S: {lr:g}", flush=True)
    for seed in SEEDS:
        pair(lr=lr, seed=seed)
    for seq_len in SEQ_LENS:
        pair(lr=lr, seq_len=seq_len)
    for size in ("M", "L"):
        pair(lr=round(lr * LR_SCALE[size], 6), size=size, save=True)
    for sinks in SINKS:
        pair(lr=lr, sinks=sinks)
    # абляция: causal-маска без диагонали, токен не видит сам себя
    pair(lr=lr, no_self=True)
