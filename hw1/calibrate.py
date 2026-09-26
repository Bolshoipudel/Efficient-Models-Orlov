import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares, nnls

from equations import bytes_moved, energy, flops, latency, memory

RESULTS = Path(__file__).resolve().parent / "results"
PEAK_FLOPS = 35.5e12
PEAK_BW = 504e9


def load():
    df = pd.read_csv(RESULTS / "measurements.csv")
    df["oom"] = df["memory_bytes"].astype(str) == "OOM"
    df["memory_bytes"] = pd.to_numeric(df["memory_bytes"], errors="coerce")
    df["is_validation"] = df["is_validation"].astype(str) == "True"
    return df


def fit_latency(s, b, t):
    def residuals(log_theta):
        return np.log(latency(s, b, np.exp(log_theta))) - np.log(t)

    lower = np.log([1e-7, 1e11, 1e10])
    upper = np.log([1e-3, PEAK_FLOPS, PEAK_BW])
    starts = [np.log([t0, p0, bw0]) for t0 in (1e-5, 1e-4) for p0 in (3e12, 2e13) for bw0 in (1e11, 4e11)]
    best = min((least_squares(residuals, x0, bounds=(lower, upper)) for x0 in starts), key=lambda r: r.cost)
    return [float(v) for v in np.exp(best.x)]


def fit_energy(s, b, e, theta):
    a = np.column_stack([latency(s, b, theta), flops(s, b), bytes_moved(s, b)])
    scale = a.max(axis=0)
    coef, _ = nnls(a / scale / e[:, None], np.ones_like(e))
    return [float(v) for v in coef / scale]


def errors(pred, meas):
    ape = np.abs(pred - meas) / meas
    return {"mape": float(ape.mean()), "median_ape": float(np.median(ape)), "max_ape": float(ape.max())}


def report(name, pred, meas, val):
    out = {"train": errors(pred[~val], meas[~val]), "validation": errors(pred[val], meas[val])}
    for split, m in out.items():
        print(f"{name:8s} {split:10s}  MAPE {m['mape']:6.1%}  median {m['median_ape']:6.1%}  max {m['max_ape']:6.1%}")
    return out


def oom_report():
    df = pd.read_csv(RESULTS / "oom.csv")
    real = (df["memory_bytes"].astype(str) == "OOM").to_numpy()
    pred = memory(df["image_size"].to_numpy(), df["batch"].to_numpy()) > df["limit_bytes"].to_numpy()
    configs = lambda m: [[int(s), int(b)] for s, b in zip(df["image_size"][m], df["batch"][m])]
    out = {
        "limit_bytes": int(df["limit_bytes"].iloc[0]),
        "real_oom": int(real.sum()),
        "predicted_oom": int(pred.sum()),
        "correct_oom": int((real & pred).sum()),
        "correct_fit": int((~real & ~pred).sum()),
        "false_alarm": configs(pred & ~real),
        "missed": configs(real & ~pred),
    }
    print(f"OOM at {out['limit_bytes'] / 2**30:.1f} GiB: real {out['real_oom']}, predicted {out['predicted_oom']}, "
          f"correct {out['correct_oom'] + out['correct_fit']}/{len(df)}, false alarm {out['false_alarm']}, missed {out['missed']}")
    return out


def main():
    df = load()
    ok = df[~df["oom"]]
    s, b, val = ok["image_size"].to_numpy(), ok["batch"].to_numpy(), ok["is_validation"].to_numpy()
    t, e, m = ok["latency_s"].to_numpy(), ok["energy_j"].to_numpy(), ok["memory_bytes"].to_numpy()
    train = ~val

    theta = fit_latency(s[train], b[train], t[train])
    theta_energy = fit_energy(s[train], b[train], e[train], theta) + theta

    print(f"theta:        t_launch={theta[0] * 1e6:.1f} us  P_eff={theta[1] / 1e12:.2f} TFLOP/s  BW_eff={theta[2] / 1e9:.0f} GB/s")
    print(f"theta_energy: P_idle={theta_energy[0]:.1f} W  e_flop={theta_energy[1] * 1e12:.2f} pJ/FLOP  e_byte={theta_energy[2] * 1e12:.1f} pJ/B")
    print(f"train points: {train.sum()}  validation points: {val.sum()}  OOM: {int(df['oom'].sum())}")

    metrics = {
        "latency": report("latency", latency(s, b, theta), t, val),
        "energy": report("energy", energy(s, b, theta_energy), e, val),
        "memory": report("memory", memory(s, b), m, val),
    }

    out = {
        "theta": {"t_launch_s": theta[0], "p_eff_flops": theta[1], "bw_eff_bytes": theta[2]},
        "theta_energy": {"p_idle_w": theta_energy[0], "e_flop_j": theta_energy[1], "e_byte_j": theta_energy[2]},
        "theta_vector": theta,
        "theta_energy_vector": theta_energy,
        "metrics": metrics,
    }
    if (RESULTS / "oom.csv").exists():
        out["oom"] = oom_report()
    (RESULTS / "theta.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
