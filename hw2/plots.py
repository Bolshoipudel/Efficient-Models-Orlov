import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sweep import LR_SCALE, LRS, RUNS, SEEDS, SEQ_LENS, SINKS, best_lr, run_name

RESULTS = RUNS.parent
FIGURES = RESULTS / "figures"
BASELINE = "#2a78d6"
XSA = "#eb6834"
FAST = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#8a8985"
GRID = "#e4e3df"
BAND = "#eeede9"
LINE = {
    False: dict(color=BASELINE, marker="s", ms=5, lw=2, label="baseline"),
    True: dict(color=XSA, marker="o", ms=5, lw=2, label="XSA"),
}
VARIANT = {
    "baseline": LINE[False],
    "xsa": LINE[True],
    "xsa_fast": dict(color=FAST, marker="^", ms=5, lw=2, label="XSA, fused"),
}
SIZE_SHADE = {"S": "#b3b2ac", "M": "#74736e", "L": INK}
SMOOTH = 4

plt.rcParams.update({
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 10,
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "legend.frameon": False,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def load(name):
    path = RUNS / name
    if not (path / "summary.json").exists():
        return None
    run = json.loads((path / "summary.json").read_text())
    run["log"] = pd.read_csv(path / "log.csv")
    run["eval"] = pd.read_csv(path / "eval.csv")
    return run


def pair(lr, size="S", seq_len=1024, sinks=0, seed=0):
    runs = [load(run_name(size, xsa, lr, seq_len, sinks, seed)) for xsa in (False, True)]
    return None if None in runs else runs


def size_lr(lr, size):
    return round(lr * LR_SCALE[size], 6)


def delta(runs, key):
    return runs[1][key] - runs[0][key]


def seed_pairs(lr):
    return [p for p in (pair(lr, seed=seed) for seed in [0] + SEEDS) if p]


def seed_noise(lr):
    deltas = [delta(p, "val_loss") for p in seed_pairs(lr)]
    return float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None


def plot_overhead():
    path = RESULTS / "overhead.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df = df[df["mode"] == "fwd+bwd"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for row, (sweep, label) in enumerate((("seq_len", "Seq length  ($d_{model}$ = 512)"), ("d_model", "$d_{model}$  (seq length = 1024)"))):
        g = df[df["sweep"] == sweep]
        series = {v: g[g["variant"] == v] for v in VARIANT}
        base = series["baseline"]
        ticks = np.arange(len(base))
        for ax, col, ylabel in ((axes[row, 0], "time_ms", "Fwd+bwd time, ms"), (axes[row, 1], "memory_mb", "Peak memory, fwd+bwd, MB")):
            for v, rows in series.items():
                ax.plot(ticks, rows[col], **VARIANT[v])
            ax.set_ylabel(ylabel)
        ax = axes[row, 2]
        for v in ("xsa", "xsa_fast"):
            for col, ls in (("time_ms", "-"), ("memory_mb", "--")):
                over = (series[v][col].to_numpy() / base[col].to_numpy() - 1) * 100
                ax.plot(ticks, over, color=VARIANT[v]["color"], marker=VARIANT[v]["marker"], ms=5, lw=2, ls=ls)
        ax.set_ylim(0, None)
        ax.set_ylabel("Overhead vs baseline, %")
        for ax in axes[row]:
            ax.set_xticks(ticks, base[sweep])
            ax.set_xlabel(label)
    axes[0, 0].legend()
    axes[0, 2].legend(handles=[plt.Line2D([], [], color=INK, lw=2, ls=ls, label=name) for ls, name in (("-", "time"), ("--", "memory"))])
    fig.suptitle("Attention + FFN block, batch 16, bf16", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / "overhead.png")
    plt.close(fig)


def plot_curves(lr):
    sizes = {size: pair(size_lr(lr, size), size=size) for size in LR_SCALE}
    sizes = {size: runs for size, runs in sizes.items() if runs}
    if not sizes:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), sharex=True)
    tops = [[], []]
    for size, runs in sizes.items():
        for xsa, run in enumerate(runs):
            log, ev = run["log"], run["eval"]
            style = dict(color=LINE[xsa]["color"], lw=1.8)
            axes[0, 0].plot(log["step"] / 1e3, log["train_loss"].rolling(SMOOTH).mean(), **style)
            axes[0, 1].plot(ev["step"] / 1e3, ev["val_loss"], marker=LINE[xsa]["marker"], ms=4, **style)
        for col, (frame, key) in enumerate((("log", "train_loss"), ("eval", "val_loss"))):
            last = runs[0][frame][key].iloc[-1]
            axes[0, col].annotate(size, (runs[0][frame]["step"].iloc[-1] / 1e3, last), xytext=(6, 0),
                                  textcoords="offset points", va="center", color=INK)
            tops[col].append(runs[0][frame][key][runs[0][frame]["step"] >= 1000].iloc[0])
            step = runs[0][frame]["step"] / 1e3
            diff = runs[1][frame][key] - runs[0][frame][key]
            if frame == "log":
                diff = diff.rolling(SMOOTH).mean()
            axes[1, col].plot(step, diff, color=SIZE_SHADE[size], lw=2, marker="o" if frame == "eval" else None, ms=4)
            axes[1, col].annotate(size, (step.iloc[-1], diff.iloc[-1]), xytext=(6, 0), textcoords="offset points", va="center", color=INK)
    for runs in seed_pairs(lr)[1:]:
        for col, (frame, key) in enumerate((("log", "train_loss"), ("eval", "val_loss"))):
            diff = runs[1][frame][key] - runs[0][frame][key]
            axes[1, col].plot(runs[0][frame]["step"] / 1e3, diff.rolling(SMOOTH).mean() if frame == "log" else diff,
                              color=SIZE_SHADE["S"], lw=0.8)
    for col, title in enumerate(("Training loss", "Validation loss")):
        axes[0, col].set_ylim(None, max(tops[col]) + 0.05)
        axes[0, col].set_title(title)
        axes[1, col].axhline(0, color=MUTED, lw=1)
        axes[1, col].set_ylim(-0.15, 0.03)
        axes[1, col].set_title(f"{title}: XSA − baseline  (below zero: XSA is better)")
        axes[1, col].set_xlabel("Training iteration (K)")
    axes[0, 0].legend(handles=[plt.Line2D([], [], **LINE[xsa]) for xsa in (False, True)])
    fig.tight_layout()
    fig.savefig(FIGURES / "loss_curves.png")
    plt.close(fig)


def plot_sweep(points, xlabel, name, noise):
    points = [(label, runs) for label, runs in points if runs]
    if len(points) < 2:
        return
    ticks = np.arange(len(points))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, key, title in zip(axes, ("train_loss", "val_loss"), ("Training loss", "Validation loss")):
        for xsa in (False, True):
            ax.plot(ticks, [runs[xsa][key] for _, runs in points], **LINE[xsa])
        ax.set_title(title)
    axes[0].legend()
    ax = axes[2]
    if noise:
        ax.axhspan(-2 * noise, 2 * noise, color=BAND, zorder=0)
        ax.text(0.02, noise, "seed noise, ±2σ", transform=ax.get_yaxis_transform(), color=MUTED, va="center", fontsize=8)
    ax.axhline(0, color=MUTED, lw=1)
    for key, ls, face, label in (("train_loss", "--", "white", "train"), ("val_loss", "-", INK, "validation")):
        ax.plot(ticks, [delta(runs, key) for _, runs in points], color=INK, ls=ls, marker="o", ms=5, mfc=face, lw=2, label=label)
    ax.legend()
    ax.set_title("XSA − baseline  (below zero: XSA is better)")
    for ax in axes:
        ax.set_xticks(ticks, [label for label, _ in points])
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{name}.png")
    plt.close(fig)


def plot_bias(lr, size):
    runs = pair(size_lr(lr, size), size=size)
    if not runs:
        return
    titles = (
        ("cos_vv", r"Average $\langle v_i, v_j \rangle$ for $i \neq j$"),
        ("a_ii", r"Average $a_{i,i}$ for $i > 1$"),
        ("cos_yv", r"Average $\langle y_i, v_i \rangle$  (XSA: before the projection)"),
    )
    layers = np.arange(len(runs[0]["bias"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, title) in zip(axes, titles):
        for xsa in (False, True):
            ax.plot(layers, [row[key] for row in runs[xsa]["bias"]], **LINE[xsa])
        ax.set_title(title)
        ax.set_xlabel("Layer index")
        ax.set_xticks(layers)
        ax.set_ylim(0, None)
    T = runs[0]["seq_len"]
    uniform = (np.sum(1 / np.arange(1, T + 1)) - 1) / (T - 1)
    axes[1].axhline(uniform, color=MUTED, lw=1, ls=":")
    axes[1].legend(handles=[plt.Line2D([], [], color=MUTED, lw=1, ls=":", label="uniform attention")], loc="upper right")
    axes[0].legend()
    fig.suptitle(f"Attention similarity bias, model {size}", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES / f"bias_{size}.png")
    plt.close(fig)


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]) + "\n"


def loss_row(label, runs):
    return [label, f"{runs[0]['train_loss']:.4f}", f"{runs[1]['train_loss']:.4f}", f"{delta(runs, 'train_loss'):+.4f}",
            f"{runs[0]['val_loss']:.4f}", f"{runs[1]['val_loss']:.4f}", f"{delta(runs, 'val_loss'):+.4f}"]


def write_tables(lr):
    loss_header = ["", "train SA", "train XSA", "Δ train", "val SA", "val XSA", "Δ val"]
    out = [f"lr baseline S: {lr:g}\n"]

    seeds = seed_pairs(lr)
    rows = [loss_row(f"seed {p[0]['seed']}", p) for p in seeds]
    if len(seeds) > 1:
        d_train, d_val = ([delta(p, key) for p in seeds] for key in ("train_loss", "val_loss"))
        rows.append(["mean ± std", "", "", f"{np.mean(d_train):+.4f} ± {np.std(d_train, ddof=1):.4f}",
                     "", "", f"{np.mean(d_val):+.4f} ± {np.std(d_val, ddof=1):.4f}"])
    out.append("### Сиды, модель S\n\n" + table(loss_header, rows))

    groups = [
        ("Размер модели", [(f"{size} (lr {size_lr(lr, size):g})", pair(size_lr(lr, size), size=size)) for size in LR_SCALE]),
        ("Learning rate, модель S", [(f"{v:g}", pair(v)) for v in sorted(LRS)]),
        ("Длина контекста, модель S", [(str(t), pair(lr, seq_len=t)) for t in sorted(SEQ_LENS + [1024])]),
        ("Attention sinks, модель S", [(str(k), pair(lr, sinks=k)) for k in [0] + SINKS]),
    ]
    for title, points in groups:
        out.append(f"### {title}\n\n" + table(loss_header, [loss_row(label, runs) for label, runs in points if runs]))

    def bias_mean(run, key):
        return f"{round(np.mean([row[key] for row in run['bias']]), 3) + 0.0:.3f}"

    rows = []
    for size in LR_SCALE:
        runs = pair(size_lr(lr, size), size=size)
        if runs:
            for label, run in zip(("SA", "XSA"), runs):
                rows.append([size, label] + [bias_mean(run, key) for key in ("cos_vv", "a_ii", "cos_yv", "cos_zv")])
    out.append("### Attention similarity bias, среднее по слоям\n\n" + table(
        ["", "", "cos(v_i, v_j)", "a_ii", "cos(y_i, v_i)", "cos(z_i, v_i)"], rows))

    base = load(run_name("S", False, lr, 1024, 0, 0))
    rows = []
    for label, xsa, no_self in (("SA", False, False), ("SA без внимания к себе", False, True),
                                ("XSA", True, False), ("XSA без внимания к себе", True, True)):
        run = load(run_name("S", xsa, lr, 1024, 0, 0, no_self))
        if run:
            rows.append([label, f"{run['train_loss']:.4f}", f"{run['val_loss']:.4f}", f"{run['val_loss'] - base['val_loss']:+.4f}",
                         bias_mean(run, "cos_vv"), bias_mean(run, "a_ii"), bias_mean(run, "cos_yv")])
    out.append("### Абляция: запрет внимания к своей позиции, модель S\n\n" + table(
        ["", "train", "val", "Δ val к SA", "cos(v_i, v_j)", "a_ii", "cos(y_i, v_i)"], rows))

    rows = []
    for size in LR_SCALE:
        runs = pair(size_lr(lr, size), size=size)
        if runs:
            sa, xsa = runs
            rows.append([size, f"{sa['params_body'] / 1e6:.1f}M", f"{sa['tokens_per_s'] / 1e3:.1f}K", f"{xsa['tokens_per_s'] / 1e3:.1f}K",
                         f"{xsa['tokens_per_s'] / sa['tokens_per_s'] - 1:+.1%}", f"{sa['peak_memory_bytes'] / 2**30:.2f}",
                         f"{xsa['peak_memory_bytes'] / 2**30:.2f}", f"{xsa['peak_memory_bytes'] / sa['peak_memory_bytes'] - 1:+.1%}"])
    out.append("### Скорость и память обучения\n\n" + table(
        ["", "параметры", "tok/s SA", "tok/s XSA", "Δ", "пик памяти SA, GiB", "пик памяти XSA, GiB", "Δ"], rows))

    path = RESULTS / "overhead.csv"
    if path.exists():
        df = pd.read_csv(path)
        rows = []
        for (sweep, d, t), g in df.groupby(["sweep", "d_model", "seq_len"], sort=False):
            cell = {(r["mode"], r["variant"]): r for _, r in g.iterrows()}
            row = [d, t]
            for mode, col, fmt in (("fwd", "time_ms", ".2f"), ("fwd+bwd", "time_ms", ".2f"), ("fwd+bwd", "memory_mb", ".0f")):
                base = cell[mode, "baseline"][col]
                row += [format(base, fmt)] + [f"{cell[mode, v][col] / base - 1:+.0%}" for v in ("xsa", "xsa_fast")]
            rows.append(row)
        out.append("### Overhead блока attention + FFN\n\n" + table(
            ["d_model", "T", "fwd, мс", "XSA", "XSA fused", "fwd+bwd, мс", "XSA", "XSA fused", "пик памяти, МБ", "XSA", "XSA fused"], rows))

        kernels = pd.read_csv(RESULTS / "overhead_kernels.csv").groupby("variant", sort=False)[["launches", "gpu_time_ms"]].sum()
        out.append("### Кернелы за один fwd+bwd блока (d_model 512, T 1024)\n\n" + table(
            ["вариант", "запусков кернелов", "GPU занята, мс"], [[v, int(r["launches"]), f"{r['gpu_time_ms']:.2f}"] for v, r in kernels.iterrows()]))

    (RESULTS / "tables.md").write_text("\n".join(out), encoding="utf-8")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    lr = best_lr()
    noise = seed_noise(lr)
    plot_overhead()
    plot_curves(lr)
    plot_sweep([(f"{v:g}", pair(v)) for v in sorted(LRS)], "Learning rate", "lr", noise)
    plot_sweep([(str(t), pair(lr, seq_len=t)) for t in sorted(SEQ_LENS + [1024])], "Seq length", "seq_len", noise)
    plot_sweep([(str(k), pair(lr, sinks=k)) for k in [0] + SINKS], "Num attn sinks", "sinks", noise)
    for size in LR_SCALE:
        plot_bias(lr, size)
    write_tables(lr)


if __name__ == "__main__":
    main()
