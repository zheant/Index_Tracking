"""Plot large-cap vs small-cap cumulative returns from Russell 3000 data.

Illustrates the regime change around 2020: from a large-cap dominated market
to a small-cap dominated market (COVID rally) and back.

Usage:
    python scripts/plot_regime_change.py
    python scripts/plot_regime_change.py --large_n 1000 --output results/regime_change.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="financial_data/russell3000")
    p.add_argument("--large_n", type=int, default=1000,
                   help="Number of top mktcap stocks considered large cap (default: 1000)")
    p.add_argument("--start_date", default="2014-01-02")
    p.add_argument("--end_date", default="2023-12-31")
    p.add_argument("--output", default="results/regime_change.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_path = REPO_ROOT / args.data_path

    print("Chargement des données...")
    returns = pd.read_csv(data_path / "returns_stocks.csv", index_col=0, parse_dates=True)
    mktcap  = pd.read_csv(data_path / "mktcap_stocks.csv",  index_col=0, parse_dates=True)

    returns = returns.loc[args.start_date:args.end_date]

    # Forward-fill mktcap monthly → daily, aligned on returns index
    mktcap_daily = mktcap.reindex(returns.index, method="ffill")

    # Keep only stocks present in both
    common = returns.columns.intersection(mktcap_daily.columns)
    returns = returns[common]
    mktcap_daily = mktcap_daily[common]

    print(f"Univers : {len(common)} stocks, {len(returns)} jours")

    # For each day: classify large vs small based on mktcap rank
    large_rets, small_rets = [], []

    for date, row_ret in returns.iterrows():
        mc = mktcap_daily.loc[date].dropna()
        valid = row_ret.reindex(mc.index).dropna()
        mc = mc.reindex(valid.index)

        ranked = mc.rank(ascending=False)
        large_stocks = ranked[ranked <= args.large_n].index
        small_stocks = ranked[ranked >  args.large_n].index

        def cap_weighted_return(stocks):
            w = mc[stocks]
            total = w.sum()
            if total == 0:
                return 0.0
            return (valid[stocks] * w / total).sum()

        large_rets.append(cap_weighted_return(large_stocks))
        small_rets.append(cap_weighted_return(small_stocks))

    large_series = pd.Series(large_rets, index=returns.index)
    small_series = pd.Series(small_rets, index=returns.index)

    # Rendements glissants sur 6 mois (~126 jours de bourse)
    window = 126
    large_rolling = (large_series + 1).rolling(window).apply(lambda x: x.prod(), raw=True) - 1
    small_rolling = (small_series + 1).rolling(window).apply(lambda x: x.prod(), raw=True) - 1

    large_cum = (large_series + 1).cumprod()
    small_cum = (small_series + 1).cumprod()

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [1, 1]})

    # --- Haut : rendements cumulés ---
    ax1.plot(large_cum, color="steelblue", linewidth=1.5, label=f"Large caps (top {args.large_n})")
    ax1.plot(small_cum, color="tomato",    linewidth=1.5, label="Small caps (reste)")
    ax1.axvline(pd.Timestamp("2020-02-19"), color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax1.axvline(pd.Timestamp("2020-03-23"), color="black", linestyle=":",  linewidth=1, alpha=0.6)
    ax1.set_title(f"Changement de régime — Russell 3000 (top {args.large_n} large caps)")
    ax1.set_ylabel("Rendement cumulé")
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    # --- Bas : rendements glissants 6 mois ---
    ax2.plot(large_rolling, color="steelblue", linewidth=1.5, label=f"Large caps (top {args.large_n})")
    ax2.plot(small_rolling, color="tomato",    linewidth=1.5, label="Small caps (reste)")
    ax2.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.4)
    ax2.axvline(pd.Timestamp("2020-02-19"), color="black", linestyle="--", linewidth=1, alpha=0.6, label="Pic pré-COVID (19 fév 2020)")
    ax2.axvline(pd.Timestamp("2020-03-23"), color="black", linestyle=":",  linewidth=1, alpha=0.6, label="Creux COVID (23 mars 2020)")
    ax2.set_ylabel("Rendement glissant 6 mois")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()

    out = REPO_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Graphique sauvegardé : {out}")


if __name__ == "__main__":
    main()
