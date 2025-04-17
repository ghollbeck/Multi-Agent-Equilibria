
"""Load simulation logs, compute metrics, and create plots."""
import json
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

LOG_DIR = pathlib.Path("logs")

def parse_log(path: pathlib.Path):
    rounds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rounds.append(json.loads(line))
    return rounds

def load_all():
    data = []
    for log in LOG_DIR.glob("log_*.jsonl"):
        meta = re.match(r"log_N(\d+)_noise([\d\.]+)_asym(True|False)_(\w+)_(\d+)\.jsonl", log.name)
        if not meta:
            continue
        N, noise, asym, matchup, seed = meta.groups()
        rounds = parse_log(log)
        for r in rounds:
            entry = {
                "N": int(N),
                "noise": float(noise),
                "asym": asym == "True",
                "matchup": matchup,
                "seed": int(seed),
                "round": r["round"],
                "avg_price": np.mean(r["prices"]),
                "profits": r["profits"],
            }
            data.append(entry)
    return pd.DataFrame(data)

def compute_metrics(df: pd.DataFrame, c: float):
    grouped = df.groupby(["N", "noise", "asym", "matchup", "seed"])
    metrics = []
    for keys, group in tqdm(grouped, desc="metrics"):
        avg_markup = ((group["avg_price"].mean() - c) / c)
        last_profits = group.sort_values("round").iloc[-1]["profits"]
        shares = np.array(last_profits) / np.sum(last_profits) if np.sum(last_profits) else np.full(len(last_profits), 1/len(last_profits))
        hhi = np.sum(shares ** 2)
        collusion_rounds = group[group["avg_price"] > c * 1.05]["round"]
        time_to_collusion = collusion_rounds.iloc[0] if not collusion_rounds.empty else np.nan
        metrics.append({
            "N": keys[0],
            "noise": keys[1],
            "asym": keys[2],
            "matchup": keys[3],
            "seed": keys[4],
            "avg_markup": avg_markup,
            "HHI": hhi,
            "time_to_collusion": time_to_collusion,
        })
    return pd.DataFrame(metrics)

def plot_heatmap(df: pd.DataFrame, metric: str, cbar_label: str):
    pivot = df.pivot_table(index="N", columns="noise", values=metric, aggfunc="mean")
    fig, ax = plt.subplots()
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("Number of Firms")
    ax.set_title(f"Average {metric}")
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    plt.show()

def t_test_vs_competitive(df: pd.DataFrame, c: float):
    group = df.groupby(["matchup"])
    results = {}
    for name, g in group:
        t_stat, p_val = stats.ttest_1samp(g["avg_price"], popmean=c)
        results[name] = (t_stat, p_val)
    return results

def main():
    c = 10.0
    df = load_all()
    metrics = compute_metrics(df, c)
    metrics.to_csv("metrics_summary.csv", index=False)
    plot_heatmap(metrics, "avg_markup", "Markup")
    tests = t_test_vs_competitive(df, c)
    print("\nStatistical tests vs competitive price:\n")
    for k, (t, p) in tests.items():
        print(f"{k}: t={t:.2f}, p={p:.3e}")

if __name__ == "__main__":
    main()
