"""Load master log, compute rich metrics, and produce figures."""
import json, pathlib, statistics, itertools, warnings, re, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

# Set up paths relative to the script directory
SCRIPT_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
LOG = SCRIPT_DIR / "logs/all_runs.jsonl"
PLOTS = SCRIPT_DIR / "plots";  PLOTS.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────
# 1. LOAD
# ──────────────────────────────────────────────────────────────
records = []
with LOG.open() as fh:
    for line in fh:
        records.append(json.loads(line))

df = pd.DataFrame(records)
if df.empty:
    raise SystemExit("No data found – did you run the experiments?")

# Helper: explode profit list into columns p0, p1, …
max_firms = df["N"].max()
for i in range(max_firms):
    df[f"profit_{i}"] = df["profits"].apply(lambda lst: lst[i] if i < len(lst) else np.nan)

# Cost lookup (from baseline markup): c_i = posted_baseline - Δ
DELTA = 0.2
for i in range(max_firms):
    df[f"cost_{i}"] = df.apply(
        lambda r: 10.0 - 2.0 if (r["asym"] and i == 0) else 10.0, axis=1
    )
df["avg_price"] = df["prices"].apply(np.mean)

# ──────────────────────────────────────────────────────────────
# 2. METRICS PER RUN
# ──────────────────────────────────────────────────────────────
rows = []
group_cols = ["N", "noise", "asym", "matchup", "seed"]
for keys, grp in tqdm(df.groupby(group_cols), desc="computing metrics"):
    N, noise, asym, matchup, seed = keys
    series_p = grp["avg_price"]
    markup = (series_p.mean() - 10.0) / 10.0

    # price volatility
    price_sd = series_p.std()

    # time to 5 % markup
    collusion_idx = grp[grp["avg_price"] > 10.0 * 1.05]["round"]
    t_coll = collusion_idx.iloc[0] if not collusion_idx.empty else np.nan

    # final profits
    last = grp.iloc[-1]
    profits = np.array(last["profits"])
    gini = 0 if profits.sum() == 0 else (np.abs(np.subtract.outer(profits, profits)).sum()
                                         / (2 * len(profits) * profits.sum()))

    rows.append(dict(
        N=N, noise=noise, asym=asym, matchup=matchup, seed=seed,
        avg_markup=markup, price_sd=price_sd, time_to_collusion=t_coll, gini=gini
    ))

metrics = pd.DataFrame(rows)
metrics.to_csv(SCRIPT_DIR / "metrics_summary.csv", index=False)

# ──────────────────────────────────────────────────────────────
# 3. PLOTS
# ──────────────────────────────────────────────────────────────
def heat(df, value, label):
    pivot = df.pivot_table(
        index="N",
        columns="noise",
        values=value,
        aggfunc=lambda x: np.nanmean(x)   # ignore NaNs
    ).sort_index()

    # if everything is NaN, skip the plot
    if np.all(np.isnan(pivot.values)):
        warnings.warn(f"All values NaN for {value}; skipping heat‑map.")
        return

    data = np.ma.masked_invalid(pivot.values)
    fig, ax = plt.subplots()
    im = ax.imshow(data, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("N firms")
    ax.set_title(label)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(label, rotation=-90, va="bottom")
    fig.savefig(PLOTS / f"{value}_heat.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

for col, lbl in [("avg_markup","Markup"), ("price_sd","Price SD"),
                 ("gini","Gini (profits)"), ("time_to_collusion","Time to Collusion")]:
    heat(metrics, col, lbl)

# ──────────────────────────────────────────────────────────────
# 4. Regression: does noise hinder collusion?
# ──────────────────────────────────────────────────────────────
reg = stats.linregress(metrics["noise"], metrics["avg_markup"])
print(f"\nOLS (avg markup ~ noise):  slope ={reg.slope:.3f},  p={reg.pvalue:.3e}")

# t-test vs competitive price
tt = stats.ttest_1samp(df["avg_price"], popmean=10.0)
print(f"\nOverall t-test against p=c: t ={tt.statistic:.2f}, p ={tt.pvalue:.3e}")

# Quick sanity‑check: how many actual LLM calls succeeded?
if "matchup" in df.columns and any(df["matchup"].isin(["llm", "mixed"])):
    total_rounds = len(df[df["matchup"].isin(["llm", "mixed"])])
    print(f"\nTotal LLM‑labelled rounds logged: {total_rounds}")
