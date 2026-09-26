import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

from calibrate import RESULTS, load
from equations import ACT_HEAD, ACT_PER_S2, BYTES_PER_VALUE, N_WEIGHTS, energy, flops, latency, memory

FIGURES = RESULTS / "figures"
MEASURED = "#2a78d6"
MODEL = "#52514e"
INK = "#0b0b0b"
MUTED = "#8a8985"
GRID = "#e4e3df"
BAND = "#eeede9"
ERROR = "#e34948"
DIVERGING = LinearSegmentedColormap.from_list("ratio", ["#2a78d6", "#f0efec", "#e34948"])

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


def metrics(theta, theta_energy):
    return [
        ("latency", "latency_s", "Latency, ms", 1e3, lambda s, b: latency(s, b, theta)),
        ("energy", "energy_j", "Energy per forward pass, J", 1.0, lambda s, b: energy(s, b, theta_energy)),
        ("memory", "memory_bytes", "Peak memory, MiB", 1 / 2**20, memory),
    ]


def point_style(val):
    return {"s": 22, "color": MEASURED, "facecolors": "white" if val else MEASURED, "linewidths": 1.2, "zorder": 3}


def legend_handles():
    return [
        Line2D([], [], color=MODEL, lw=2, label="model"),
        Line2D([], [], ls="", marker="o", ms=5, color=MEASURED, label="measured, train"),
        Line2D([], [], ls="", marker="o", ms=5, mfc="white", mec=MEASURED, mew=1.2, label="measured, validation"),
    ]


def plot_vs_batch(df, key, col, label, scale, fn, val_sizes):
    sizes = sorted(df["image_size"].unique())
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), sharex=True, sharey=True)
    dense_b = np.arange(1, 257)
    for ax, s in zip(axes.flat, sizes):
        g = df[df["image_size"] == s]
        ax.plot(dense_b, fn(s, dense_b) * scale, color=MODEL, lw=2, zorder=2)
        for val in (False, True):
            h = g[g["is_validation"] == val]
            ax.scatter(h["batch"], h[col] * scale, **point_style(val))
        ax.set_title(f"S = {s}" + (" (validation S)" if s in val_sizes else ""))
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
    last = axes.flat[-1]
    last.axis("off")
    last.legend(handles=legend_handles(), loc="center")
    for ax in axes[-1]:
        ax.set_xlabel("Batch size B")
    for ax in axes[:, 0]:
        ax.set_ylabel(label)
    fig.suptitle(f"{key.capitalize()} vs batch size, per image size S", color=INK, fontsize=12)
    fig.savefig(FIGURES / f"{key}_vs_batch.png")
    plt.close(fig)


def plot_ratio_maps(df, specs, val_sizes, val_batches):
    sizes = sorted(df["image_size"].unique())
    batches = sorted(df["batch"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    lim = 2.0
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    for ax, (key, col, label, scale, fn) in zip(axes, specs):
        grid = np.full((len(batches), len(sizes)), np.nan)
        for _, r in df.iterrows():
            i, j = batches.index(r["batch"]), sizes.index(r["image_size"])
            grid[i, j] = np.log2(r[col] / fn(r["image_size"], r["batch"]))
        im = ax.imshow(grid, cmap=DIVERGING, norm=norm, origin="lower", aspect="auto")
        for i in range(len(batches)):
            for j in range(len(sizes)):
                v = grid[i, j]
                if np.isnan(v):
                    ax.text(j, i, "OOM", ha="center", va="center", fontsize=6, color=MUTED)
                    continue
                ax.text(j, i, f"{(2**v - 1) * 100:+.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 * lim else INK)
        ax.set_xticks(range(len(sizes)), [f"{s}*" if s in val_sizes else str(s) for s in sizes], rotation=45)
        ax.set_yticks(range(len(batches)), [f"{b}*" if b in val_batches else str(b) for b in batches])
        ax.set_xlabel("Image size S  (* = validation value)")
        ax.set_ylabel("Batch size B  (* = validation value)")
        ax.set_title(f"{key.capitalize()}: measured vs predicted, %")
        ax.grid(False)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    ticks = np.log2([0.25, 0.5, 1, 2, 4])
    cbar.set_ticks(ticks, labels=["×0.25", "×0.5", "×1", "×2", "×4"])
    cbar.set_label("measured / predicted  (red = model underestimates)")
    fig.savefig(FIGURES / "ratio_maps.png")
    plt.close(fig)


def plot_throughput(df, theta):
    s, b = df["image_size"].to_numpy(), df["batch"].to_numpy()
    work = flops(s, b)
    achieved = work / df["latency_s"].to_numpy() / 1e12
    ds, db = np.meshgrid(np.arange(32, 513, 16), np.arange(1, 257))
    dwork = flops(ds, db).ravel()
    order = np.argsort(dwork)
    dmodel = (dwork / latency(ds, db, theta).ravel() / 1e12)[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(dwork[order], dmodel, color=MODEL, lw=2, zorder=2, label="model")
    ax.axhline(theta[1] / 1e12, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(dwork.min(), theta[1] / 1e12 * 1.06, f"fitted P_eff = {theta[1] / 1e12:.1f} TFLOP/s", color=INK, fontsize=8)
    for val in (False, True):
        m = df["is_validation"].to_numpy() == val
        ax.scatter(work[m], achieved[m], **point_style(val), label="measured, validation" if val else "measured, train")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Work per forward pass, FLOPs")
    ax.set_ylabel("Achieved throughput, TFLOP/s")
    ax.set_title("Achieved throughput: launch-bound (left) to compute/memory-bound (right)")
    ax.legend(loc="lower right")
    fig.savefig(FIGURES / "throughput.png")
    plt.close(fig)


def plot_oom(df):
    limit = df["limit_bytes"].iloc[0]
    real = (df["memory_bytes"].astype(str) == "OOM").to_numpy()
    s, b = df["image_size"].to_numpy(), df["batch"].to_numpy()
    pred = memory(s, b) > limit
    wrong = real != pred

    fig, ax = plt.subplots(figsize=(10, 5.5))
    dense_s = np.linspace(32, 512, 400)
    b_max = (limit / BYTES_PER_VALUE - N_WEIGHTS) / (ACT_PER_S2 * dense_s**2 + ACT_HEAD)
    ax.fill_between(dense_s, b_max, 400, color=BAND, lw=0, zorder=1)
    ax.plot(dense_s, b_max, color=MODEL, lw=2, zorder=2)
    for is_oom, marker in ((False, "o"), (True, "X")):
        for is_wrong, color in ((False, MUTED), (True, ERROR)):
            m = (real == is_oom) & (wrong == is_wrong)
            ax.scatter(s[m], b[m], marker=marker, s=46 if is_oom else 30, color=color, zorder=3,
                       edgecolors="white", linewidths=0.8)
    for si, bi, r in zip(s[wrong], b[wrong], real[wrong]):
        ax.annotate("missed" if r else "false alarm", (si, bi), xytext=(-9, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(sorted(df["image_size"].unique()), [str(v) for v in sorted(df["image_size"].unique())], rotation=60, ha="right", rotation_mode="anchor")
    ax.set_yticks(sorted(df["batch"].unique()), [str(v) for v in sorted(df["batch"].unique())])
    ax.minorticks_off()
    ax.set_xlim(28, 580)
    ax.set_ylim(0.8, 330)
    ax.set_xlabel("Image size S")
    ax.set_ylabel("Batch size B")
    ax.set_title(f"OOM with the allocator limited to {limit / 2**30:.0f} GiB: measured vs predicted")
    handles = [
        Line2D([], [], color=MODEL, lw=2, label="model boundary: Memory(S, B) = limit"),
        plt.Rectangle((0, 0), 1, 1, color=BAND, label="model predicts OOM"),
        Line2D([], [], ls="", marker="o", color=MUTED, label="measured: fits"),
        Line2D([], [], ls="", marker="X", ms=8, color=MUTED, label="measured: OOM"),
        Line2D([], [], ls="", marker="o", color=ERROR, label="model is wrong"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.savefig(FIGURES / "oom.png")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    params = json.loads((RESULTS / "theta.json").read_text())
    theta, theta_energy = params["theta_vector"], params["theta_energy_vector"]
    df = load()
    ok = df[~df["oom"]]
    val_sizes = {s for s, g in df.groupby("image_size") if g["is_validation"].all()}
    val_batches = {b for b, g in df.groupby("batch") if g["is_validation"].all()}
    specs = metrics(theta, theta_energy)

    for spec in specs:
        plot_vs_batch(ok, *spec, val_sizes)
    if (RESULTS / "oom.csv").exists():
        plot_oom(pd.read_csv(RESULTS / "oom.csv"))
    plot_ratio_maps(df, specs, val_sizes, val_batches)
    plot_throughput(ok, theta)
    print("saved:", ", ".join(p.name for p in sorted(FIGURES.glob("*.png"))))


if __name__ == "__main__":
    main()
