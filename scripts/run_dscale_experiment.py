"""Automated d_scale sweep for quob_stratified on Russell 3000.

For each d_scale value:
  1. Runs main.py with the given parameters and saves the PKL.
  2. Renames the PKL to include a d_scale tag so subsequent runs don't overwrite it.

Then loads all tagged PKLs, reconstructs out-of-sample returns via extract_timeseries,
and produces standard comparison plots plus a summary bar chart of key metrics.

Usage (full sweep):
    python scripts/run_dscale_experiment.py \\
        --d_scales 0.5 1.0 2.0 5.0 10.0 \\
        --replicator_cores 64 --time_limit 225

Usage (analysis only, PKLs already generated):
    python scripts/run_dscale_experiment.py \\
        --d_scales 0.5 1.0 2.0 5.0 10.0 --skip_runs
"""
from __future__ import annotations

import argparse
import subprocess
import shutil
import sys
from argparse import Namespace
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kurtosis, skew

# Repo root (parent of scripts/) for prafa/ imports
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))  # so we can import analyze_results as a sibling module

from analyze_results import (  # noqa: E402
    extract_timeseries,
    _plot_cumulative,
    _plot_tracking_errors,
    _plot_mae,
    _plot_combined,
    _plot_error_distributions,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep d_scale values for quob_stratified.")
    p.add_argument("--d_scales", nargs="+", type=float, default=[0.5, 1.0, 2.0, 5.0, 10.0],
                   help="List of d_scale values to test (default: 0.5 1.0 2.0 5.0 10.0).")
    p.add_argument("--skip_runs", action="store_true",
                   help="Skip main.py runs; only (re)analyze existing PKLs.")

    # Parameters forwarded to main.py
    p.add_argument("--index", default="russell3000")
    p.add_argument("--cardinality", type=int, default=300)
    p.add_argument("--solution_name", default="quob_stratified")
    p.add_argument("--start_date", default="2014-01-02")
    p.add_argument("--end_date", default="2023-12-31")
    p.add_argument("--rebalancing", type=int, default=6)
    p.add_argument("--min_trading_frac", type=float, default=0.20)
    p.add_argument("--max_missing_frac", type=float, default=0.10)
    p.add_argument("--winsor_sigma", type=float, default=3.0)
    p.add_argument("--hard_clip", type=float, default=1.0)
    p.add_argument("--distance_method", default="pearson")
    p.add_argument("--missing_policy", default="strict")
    p.add_argument("--replicator_cores", type=int, default=8)
    p.add_argument("--time_limit", type=float, default=225)
    p.add_argument("--strata_large_size", type=int, default=1000)

    # Paths
    p.add_argument("--data_path", default="financial_data")
    p.add_argument("--result_path", default="results")
    p.add_argument("--output_dir", default=None,
                   help="Where to write plots (default: results/dscale_experiment).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# PKL naming helpers
# ---------------------------------------------------------------------------

def _pkl_suffix(args: argparse.Namespace) -> str:
    """Mirror the suffix logic in Portfolio.save_portfolio()."""
    if getattr(args, 'exclude_pool_b_capweight', False):
        return "_phase16_pool_a_only"
    if getattr(args, 'phase18_qp_index', False):
        return "_phase18_qp_index"
    if getattr(args, 'min_trading_frac', 0.50) == 0.0 and getattr(args, 'no_stratification', False):
        return "_phase17_no_strat"
    if getattr(args, 'min_trading_frac', 0.50) == 0.0:
        return "_phase17_no_liq_filter"
    return ""


def tagged_pkl(args: argparse.Namespace, d_scale: float) -> Path:
    """PKL path including the d_scale tag."""
    suffix = _pkl_suffix(args)
    tag = f"dscale_{d_scale:g}"
    return Path(args.result_path) / (
        f"portfolio_{args.index}_{args.solution_name}_{args.cardinality}{suffix}_{tag}.pkl"
    )


def default_pkl(args: argparse.Namespace) -> Path:
    """Path where main.py writes the PKL before we rename it."""
    suffix = _pkl_suffix(args)
    return Path(args.result_path) / (
        f"portfolio_{args.index}_{args.solution_name}_{args.cardinality}{suffix}.pkl"
    )


# ---------------------------------------------------------------------------
# Run main.py for one d_scale value
# ---------------------------------------------------------------------------

def run_one(args: argparse.Namespace, d_scale: float) -> None:
    target = tagged_pkl(args, d_scale)
    if target.exists():
        print(f"[d_scale={d_scale:g}] PKL already exists, skipping run: {target.name}")
        return

    cmd = [
        sys.executable, str(REPO_ROOT / "main.py"),
        "--index", args.index,
        "--cardinality", str(args.cardinality),
        "--solution_name", args.solution_name,
        "--start_date", args.start_date,
        "--end_date", args.end_date,
        "--rebalancing", str(args.rebalancing),
        "--min_trading_frac", str(args.min_trading_frac),
        "--max_missing_frac", str(args.max_missing_frac),
        "--winsor_sigma", str(args.winsor_sigma),
        "--hard_clip", str(args.hard_clip),
        "--distance_method", args.distance_method,
        "--missing_policy", args.missing_policy,
        "--replicator_cores", str(args.replicator_cores),
        "--time_limit", str(args.time_limit),
        "--strata_large_size", str(args.strata_large_size),
        "--d_scale", str(d_scale),
    ]

    sep = "=" * 64
    print(f"\n{sep}\n[d_scale={d_scale:g}] Launching run\n{sep}\n")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    src = default_pkl(args)
    if not src.exists():
        raise FileNotFoundError(f"Expected PKL not found after run: {src}")
    shutil.move(str(src), str(target))
    print(f"[d_scale={d_scale:g}] PKL saved -> {target.name}")


# ---------------------------------------------------------------------------
# Build the args namespace expected by Universe / extract_timeseries
# ---------------------------------------------------------------------------

def build_analysis_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        index=args.index,
        data_path=args.data_path,
        result_path=args.result_path,
        solution_name=args.solution_name,
        rebalancing=args.rebalancing,
        cardinality=args.cardinality,
        start_date=args.start_date,
        end_date=args.end_date,
        missing_policy=(
            args.missing_policy if args.missing_policy != "auto"
            else ("legacy" if args.index == "sp500" else "strict")
        ),
        min_presence=0.90,
        reconstitution_month=7,
        max_missing_frac=args.max_missing_frac,
        min_trading_frac=args.min_trading_frac,
        winsor_sigma=args.winsor_sigma,
        hard_clip=args.hard_clip,
    )


# ---------------------------------------------------------------------------
# Summary bar chart
# ---------------------------------------------------------------------------

def plot_summary(metrics: Dict[float, dict], output_dir: Path) -> None:
    d_scales = sorted(metrics)
    labels = [f"{d:g}" for d in d_scales]
    x = list(range(len(d_scales)))

    means     = [metrics[d]["mean"]     for d in d_scales]
    variances = [metrics[d]["variance"] for d in d_scales]
    mean_te   = [metrics[d]["mean_te"]  for d in d_scales]
    mean_mae  = [metrics[d]["mean_mae"] for d in d_scales]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].bar(x, means, color="steelblue")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("Biais moyen journalier (r_p - r_i)")
    axes[0].set_xlabel("d_scale")

    axes[1].bar(x, variances, color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Variance de l'erreur de replication")
    axes[1].set_xlabel("d_scale")

    axes[2].bar([xi - 0.2 for xi in x], mean_te,  width=0.4, color="mediumseagreen", label="TE moyen")
    axes[2].bar([xi + 0.2 for xi in x], mean_mae, width=0.4, color="mediumpurple",   label="MAE moyen")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_title("Tracking Error & MAE moyens par fenetre")
    axes[2].set_xlabel("d_scale")
    axes[2].legend()

    plt.suptitle("Impact de d_scale — quob_stratified Russell 3000", fontsize=13, y=1.02)
    plt.tight_layout()
    out = output_dir / "dscale_summary_metrics.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Resume sauvegarde : {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(args.result_path) / "dscale_experiment"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: runs -------------------------------------------------------
    if not args.skip_runs:
        for d in args.d_scales:
            run_one(args, d)

    # ---- Step 2: analysis ---------------------------------------------------
    base_args = build_analysis_args(args)

    rendements: dict = {}
    tracking_errors_all: dict = {}
    mae_all: dict = {}
    indice_reference = None
    summary_metrics: Dict[float, dict] = {}

    for d in sorted(args.d_scales):
        path = tagged_pkl(args, d)
        if not path.exists():
            print(f"[d_scale={d:g}] PKL not found, skipping analysis: {path.name}")
            continue

        label = f"d_scale={d:g}"
        print(f"\n{'─'*50}\nAnalyse : {label}  ({path.name})\n{'─'*50}")

        rp, ri, te, mae = extract_timeseries(path, base_args)
        rendements[label] = rp
        tracking_errors_all[label] = te
        mae_all[label] = mae
        if indice_reference is None:
            indice_reference = ri

        erreurs = (rp - ri).dropna()
        summary_metrics[d] = {
            "mean":     float(erreurs.mean()),
            "variance": float(erreurs.var()),
            "skewness": float(skew(erreurs)),
            "kurtosis": float(kurtosis(erreurs)),
            "mean_te":  float(pd.Series(list(te.values())).mean()),
            "mean_mae": float(pd.Series(list(mae.values())).mean()),
        }

    if indice_reference is None:
        print("Aucun PKL trouve. Verifiez que les runs ont bien ete executes.")
        return

    # ---- Step 3: plots ------------------------------------------------------
    _plot_cumulative(rendements, indice_reference, output_dir)
    _plot_tracking_errors(tracking_errors_all, output_dir)
    _plot_mae(mae_all, output_dir)
    _plot_combined(rendements, indice_reference, output_dir)
    _plot_error_distributions(rendements, indice_reference, output_dir)
    plot_summary(summary_metrics, output_dir)

    # ---- Step 4: console summary --------------------------------------------
    sep = "=" * 72
    print(f"\n{sep}")
    print(
        f"{'d_scale':>10} {'Biais':>10} {'Variance':>12} "
        f"{'Skew':>8} {'Kurt':>8} {'TE moy':>10} {'MAE moy':>10}"
    )
    print("-" * 72)
    for d, m in sorted(summary_metrics.items()):
        print(
            f"{d:>10g} {m['mean']:>10.5f} {m['variance']:>12.6f} "
            f"{m['skewness']:>8.3f} {m['kurtosis']:>8.3f} "
            f"{m['mean_te']:>10.5f} {m['mean_mae']:>10.5f}"
        )
    print(sep)
    print(f"\nGraphiques dans : {output_dir}")


if __name__ == "__main__":
    main()
