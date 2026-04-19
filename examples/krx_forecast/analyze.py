"""Analyze Kronos-base backtest results on KRX universe and produce summary visuals.

Reads:
    output/metrics.csv
    output/backtest/{code}.csv    (predicted vs actual close over holdout)
    output/forward/{code}.csv     (next-60-day forecast)

Writes:
    output/analysis/summary.csv               - cleaned, ranked metrics
    output/analysis/top_best.csv              - top 20 by MAPE (tie-break DirAcc)
    output/analysis/top_worst.csv             - bottom 20 by MAPE
    output/analysis/overview.png              - 4-panel distribution + market split
    output/analysis/best_predictions.png      - actual vs pred close for 6 best
    output/analysis/worst_predictions.png     - actual vs pred close for 6 worst
    output/analysis/forward_showcase.png      - 6 interesting forward forecasts
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "output"
ANALYSIS = OUT / "analysis"
ANALYSIS.mkdir(parents=True, exist_ok=True)

import matplotlib.font_manager as fm

_KR_FONT_CANDIDATES = ["Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim", "Batang"]
_available = {f.name for f in fm.fontManager.ttflist}
_chosen = next((f for f in _KR_FONT_CANDIDATES if f in _available), "DejaVu Sans")

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.family": _chosen,
    "axes.unicode_minus": False,
})


def load_metrics() -> pd.DataFrame:
    m = pd.read_csv(OUT / "metrics.csv", dtype={"code": str})
    m["code"] = m["code"].str.zfill(6)
    # A holdout window with essentially flat close produces degenerate dir_acc (0.0 or 1.0)
    # and MAPE inflated by divide-by-near-zero. Flag those so they don't skew summaries.
    m["degenerate"] = (m["dir_acc"] == 0.0) | (m["dir_acc"] == 1.0) | (m["mape_close_pct"] > 200)
    return m


def plot_overview(m: pd.DataFrame) -> None:
    clean = m[~m["degenerate"]]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.hist(clean["mape_close_pct"], bins=60, range=(0, 80), color="#4C72B0", edgecolor="white")
    ax.axvline(clean["mape_close_pct"].median(), color="red", linestyle="--",
               label=f"median={clean['mape_close_pct'].median():.1f}%")
    ax.set_xlabel("MAPE (%) on close, 60-day holdout")
    ax.set_ylabel("# stocks")
    ax.set_title(f"MAPE distribution (n={len(clean)})")
    ax.legend()

    ax = axes[0, 1]
    ax.hist(clean["dir_acc"], bins=40, range=(0.2, 0.8), color="#55A868", edgecolor="white")
    ax.axvline(0.5, color="black", linestyle=":", label="coin flip (0.5)")
    ax.axvline(clean["dir_acc"].mean(), color="red", linestyle="--",
               label=f"mean={clean['dir_acc'].mean():.3f}")
    ax.set_xlabel("Directional accuracy (day-over-day sign)")
    ax.set_ylabel("# stocks")
    ax.set_title("Directional accuracy distribution")
    ax.legend()

    ax = axes[1, 0]
    for mkt, sub in clean.groupby("market"):
        ax.hist(sub["mape_close_pct"], bins=60, range=(0, 80), alpha=0.55,
                label=f"{mkt} (n={len(sub)})", edgecolor="white")
    ax.set_xlabel("MAPE (%)")
    ax.set_ylabel("# stocks")
    ax.set_title("MAPE by market")
    ax.legend()

    ax = axes[1, 1]
    sample = clean.sample(n=min(1500, len(clean)), random_state=0)
    colors = {"KOSPI": "#C44E52", "KOSDAQ": "#4C72B0"}
    for mkt, sub in sample.groupby("market"):
        ax.scatter(sub["mape_close_pct"], sub["dir_acc"], s=6, alpha=0.35,
                   c=colors.get(mkt, "grey"), label=mkt)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax.set_xlim(0, 80)
    ax.set_ylim(0.2, 0.8)
    ax.set_xlabel("MAPE (%)")
    ax.set_ylabel("Directional accuracy")
    ax.set_title("MAPE vs DirAcc (random 1.5k sample)")
    ax.legend(markerscale=2)

    plt.tight_layout()
    plt.savefig(ANALYSIS / "overview.png", bbox_inches="tight")
    plt.close()


def _plot_grid(codes_names: list, title: str, out_path: Path, folder: str) -> None:
    """Plot 6 ticker close curves (actual vs pred for backtest, pred only for forward)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    for ax, (code, name) in zip(axes.flat, codes_names):
        path = OUT / folder / f"{code}.csv"
        if not path.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(path, parse_dates=["timestamps"])
        if folder == "backtest":
            ax.plot(df["timestamps"], df["actual_close"], color="#2C3E50", linewidth=1.6, label="actual")
            ax.plot(df["timestamps"], df["pred_close"], color="#E74C3C", linewidth=1.6, label="pred")
        else:
            ax.plot(df["timestamps"], df["close"], color="#E74C3C", linewidth=1.6, label="pred")
        ax.set_title(f"{code} {name}", fontsize=11)
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.legend(loc="best", fontsize=9)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    m = load_metrics()
    clean = m[~m["degenerate"]].copy()

    summary = m.sort_values(["degenerate", "mape_close_pct"]).reset_index(drop=True)
    summary.to_csv(ANALYSIS / "summary.csv", index=False)

    best = clean.sort_values(["mape_close_pct", "dir_acc"], ascending=[True, False]).head(20)
    worst = clean.sort_values(["mape_close_pct", "dir_acc"], ascending=[False, True]).head(20)
    best.to_csv(ANALYSIS / "top_best.csv", index=False)
    worst.to_csv(ANALYSIS / "top_worst.csv", index=False)

    print("=== Summary ===")
    print(f"  total tickers: {len(m)}")
    print(f"  degenerate (flat holdout / extreme MAPE): {int(m['degenerate'].sum())}")
    print(f"  clean: {len(clean)}")
    print()
    print("  MAPE (close):   mean={:.2f}%  median={:.2f}%  p25={:.2f}%  p75={:.2f}%".format(
        clean["mape_close_pct"].mean(), clean["mape_close_pct"].median(),
        clean["mape_close_pct"].quantile(0.25), clean["mape_close_pct"].quantile(0.75)))
    print("  DirAcc:         mean={:.3f}  median={:.3f}  p25={:.3f}  p75={:.3f}".format(
        clean["dir_acc"].mean(), clean["dir_acc"].median(),
        clean["dir_acc"].quantile(0.25), clean["dir_acc"].quantile(0.75)))
    better = (clean["dir_acc"] > 0.5).sum()
    print(f"  DirAcc > 0.5 (beating coin flip): {better}/{len(clean)} = {better/len(clean)*100:.1f}%")

    by_mkt = clean.groupby("market").agg(
        n=("code", "size"),
        mape_mean=("mape_close_pct", "mean"),
        mape_median=("mape_close_pct", "median"),
        dir_acc_mean=("dir_acc", "mean"),
        beat_50_pct=("dir_acc", lambda s: (s > 0.5).mean() * 100),
    ).round(3)
    print("\n=== By market ===")
    print(by_mkt.to_string())

    print("\n=== Top 10 best (lowest MAPE among non-degenerate) ===")
    print(best.head(10)[["code", "name", "market", "mape_close_pct", "dir_acc"]].to_string(index=False))
    print("\n=== Top 10 worst ===")
    print(worst.head(10)[["code", "name", "market", "mape_close_pct", "dir_acc"]].to_string(index=False))

    # Visuals
    plot_overview(m)
    best6 = best.head(6)[["code", "name"]].values.tolist()
    worst6 = worst.head(6)[["code", "name"]].values.tolist()
    _plot_grid(best6, "Best 6 backtests (low MAPE)", ANALYSIS / "best_predictions.png", "backtest")
    _plot_grid(worst6, "Worst 6 backtests (high MAPE)", ANALYSIS / "worst_predictions.png", "backtest")

    # Forward showcase: top-6 by dir_acc high + mape low (a "trust-worthy" proxy)
    trustworthy = clean.copy()
    trustworthy["score"] = trustworthy["dir_acc"] - trustworthy["mape_close_pct"] / 100.0
    show = trustworthy.sort_values("score", ascending=False).head(6)[["code", "name"]].values.tolist()
    _plot_grid(show, "Forward forecast (next 60 business days) - 6 trustworthy tickers",
               ANALYSIS / "forward_showcase.png", "forward")

    print("\nOutputs written to:", ANALYSIS)
    for p in sorted(ANALYSIS.iterdir()):
        print(" ", p.name)


if __name__ == "__main__":
    main()
