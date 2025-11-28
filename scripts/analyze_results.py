"""Replicate the analyses_resultats.ipynb methodology for saved portfolios.

This script rebuilds the out-of-sample return series for one or more
solution methods, using the same logic as the notebook:
- load saved portfolios (pickled dict keyed by rebalance date)
- roll forward between consecutive rebalances to compute portfolio vs. index returns
- compute tracking error (std dev) and mean absolute error per window
- plot cumulative returns, tracking error scatter, absolute error scatter,
  cumulative returns + absolute deviations, and error distributions.
"""
from __future__ import annotations

import argparse
import pickle
from argparse import Namespace
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")  # ensure headless rendering
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis, skew

# Ensure the repository root (containing the prafa package) is on sys.path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from prafa.universe import Universe


PortfolioDict = Dict[pd.Timestamp, pd.Series]
ReturnSeries = pd.Series


def _load_portfolios(path: Path) -> PortfolioDict:
    if not path.exists():
        raise FileNotFoundError(f"Portfolio file not found: {path}")

    with path.open("rb") as f:
        portfolios = pickle.load(f)

    if not isinstance(portfolios, dict):
        raise ValueError(f"Unexpected portfolio format in {path}; expected dict of weights by rebalance date.")
    return portfolios


def _align_weights(weights, columns: Iterable[str], target_cardinality: int | None = None) -> pd.Series:
    series = pd.Series(weights)
    target_cols = list(columns)
    if series.index.dtype == "int64":
        # If weights came from a numpy array, align to provided columns order
        if len(series) != len(target_cols):
            print(
                f"⚠️ Weight length mismatch (weights: {len(series)}, columns: {len(target_cols)}); "
                "truncating/padding to match cleaned universe."
            )
        series = pd.Series(series.values[: len(target_cols)], index=target_cols[: len(series)])
    series = series.reindex(target_cols, fill_value=0)

    non_zero = (series != 0).sum()
    if target_cardinality is not None and non_zero < target_cardinality:
        print(
            f"⚠️ Effective invested names after alignment ({non_zero}) below target cardinality ({target_cardinality})."
        )

    total_weight = series.sum()
    if total_weight > 0:
        series = series / total_weight
    else:
        print("⚠️ Aligned weights sum to zero; portfolio will have no exposure.")
    return series


def extract_timeseries(filepath: Path, base_args: argparse.Namespace) -> Tuple[ReturnSeries, ReturnSeries, Dict[pd.Timestamp, float], Dict[pd.Timestamp, float]]:
    portfolios = _load_portfolios(filepath)

    args = Namespace(**vars(base_args))
    universe = Universe(args)

    # Ensure dates are sorted to iterate consecutive rebalancing windows
    dates = sorted(portfolios.keys())
    n = len(dates)
    if n < 2:
        raise ValueError("At least two rebalance dates are required to compute out-of-sample returns.")

    rendements_portefeuille = []
    rendements_indice = []
    index_dates = []
    tracking_errors: Dict[pd.Timestamp, float] = {}
    mae: Dict[pd.Timestamp, float] = {}

    for i in range(n - 1):
        start_date = pd.Timestamp(dates[i])
        # veille du prochain rebalance
        end_date = pd.Timestamp(dates[i + 1]) - pd.tseries.offsets.BDay(1)

        universe.new_universe(start_date, end_date, training=False)
        X_test = universe.get_stocks_returns()
        Y_test = universe.get_index_returns()
        weights = portfolios[start_date]
        weights_series = _align_weights(weights, X_test.columns, target_cardinality=args.cardinality)

        if X_test.shape[1] < args.cardinality:
            print(
                f"⚠️ Cleaned universe size ({X_test.shape[1]}) below target cardinality ({args.cardinality}) for window starting {start_date.date()}."
            )

        assert all(X_test.index == Y_test.index), "Les index de X_test et Y_test ne sont pas alignés !"

        return_outsample = X_test @ weights_series
        tracking_error = (return_outsample - Y_test).std()
        tracking_errors[X_test.index[-1]] = tracking_error
        mae[X_test.index[-1]] = (return_outsample - Y_test).abs().mean()

        rendements_portefeuille += list(return_outsample)
        rendements_indice += list(Y_test)
        index_dates += list(X_test.index)

    rendements_portefeuille = pd.Series(rendements_portefeuille, index=index_dates)
    rendements_indice = pd.Series(rendements_indice, index=index_dates)
    return rendements_portefeuille, rendements_indice, tracking_errors, mae


def _plot_cumulative(rendements: Dict[str, ReturnSeries], indice_reference: ReturnSeries, output_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    for method, rp in rendements.items():
        plt.plot((rp + 1).cumprod(), label=f"Portefeuille – {method}")
    plt.plot((indice_reference + 1).cumprod(), label="Indice", color="black")
    plt.title("Rendements cumulés")
    plt.xlabel("Date")
    plt.ylabel("Rendement cumulé")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_returns.png", bbox_inches="tight")
    plt.close()


def _plot_tracking_errors(tracking_errors_all: Dict[str, Dict[pd.Timestamp, float]], output_dir: Path) -> None:
    plt.figure(figsize=(10, 4))
    for method, te in tracking_errors_all.items():
        dates = list(te.keys())
        values = list(te.values())
        plt.scatter(dates, values, label=f"Tracking Error – {method}", s=25)
    plt.title("Tracking Errors")
    plt.xlabel("Date")
    plt.ylabel("Écart-type (Tracking Error)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "tracking_errors.png", bbox_inches="tight")
    plt.close()


def _plot_mae(mae_all: Dict[str, Dict[pd.Timestamp, float]], output_dir: Path) -> None:
    plt.figure(figsize=(10, 4))
    for method, ae in mae_all.items():
        dates = list(ae.keys())
        values = list(ae.values())
        plt.scatter(dates, values, label=f"Tracking Absolute Error – {method}", s=25)
    plt.title("Tracking Absolute Error")
    plt.xlabel("Date")
    plt.ylabel("Écart-type (Tracking Error)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "tracking_absolute_errors.png", bbox_inches="tight")
    plt.close()


def _plot_combined(rendements: Dict[str, ReturnSeries], indice_reference: ReturnSeries, output_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    for method, rp in rendements.items():
        ax1.plot((rp + 1).cumprod(), label=f"Portefeuille – {method}")
    ax1.plot((indice_reference + 1).cumprod(), label="Indice", color="black")
    ax1.set_title("Rendements cumulés")
    ax1.set_ylabel("Rendement cumulé")
    ax1.legend()
    ax1.grid(True)

    for method, rp in rendements.items():
        ecarts_absolus = rp - indice_reference
        ax2.plot(rp.index, ecarts_absolus, label=f"Écarts absolus – {method}", alpha=0.5)
    ax2.set_title("Écarts absolus à chaque pas de temps")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Écart absolu")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "cumulative_and_absolute_errors.png", bbox_inches="tight")
    plt.close()


def _plot_error_distributions(rendements: Dict[str, ReturnSeries], indice_reference: ReturnSeries, output_dir: Path) -> None:
    fig, axes = plt.subplots(nrows=len(rendements), ncols=1, figsize=(13, 4 * len(rendements)), sharex=True)
    if len(rendements) == 1:
        axes = [axes]

    for ax, (method, rp) in zip(axes, rendements.items()):
        erreurs = rp - indice_reference
        moyenne = erreurs.mean()
        mediane = erreurs.median()
        variance = erreurs.var()
        skewness = skew(erreurs)
        kurt = kurtosis(erreurs)

        sns.histplot(erreurs, kde=True, bins=200, ax=ax, color="skyblue", edgecolor="black")
        ax.axvline(moyenne, color="red", linestyle="--", linewidth=1.5, label=f"Moyenne: {moyenne:.5f}")
        ax.axvline(mediane, color="green", linestyle="--", linewidth=1.5, label=f"Médiane: {mediane:.5f}")

        ax.set_title(f"Distribution des erreurs de réplication – {method}", fontsize=12)
        ax.set_ylabel("Densité")
        ax.legend()

        textstr = "\n".join(
            (
                f"Variance : {variance:.6f}",
                f"Skewness : {skewness:.3f}",
                f"Kurtosis : {kurt:.3f}",
            )
        )
        ax.text(
            0.98,
            0.95,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        )

    axes[-1].set_xlabel("Erreur de réplication (r_portefeuille - r_indice)")
    plt.tight_layout()
    plt.savefig(output_dir / "error_distributions.png", bbox_inches="tight")
    plt.close()


def _build_args(cli_args: argparse.Namespace, solution_name: str) -> argparse.Namespace:
    return Namespace(
        index=cli_args.index,
        data_path=cli_args.data_path,
        result_path=cli_args.result_path,
        solution_name=solution_name,
        rebalancing=cli_args.rebalancing,
        cardinality=cli_args.cardinality,
        start_date=cli_args.start_date,
        end_date=cli_args.end_date,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce analyses_resultats.ipynb charts for saved portfolios.")
    parser.add_argument("--index", default="russell3000", help="Index name used when generating the portfolios (default: russell3000).")
    parser.add_argument("--data_path", default="financial_data", help="Base folder for financial data (default: financial_data).")
    parser.add_argument("--result_path", default="results", help="Folder containing saved portfolio pickles (default: results).")
    parser.add_argument("--solutions", nargs="+", default=["quob", "gurobi"], help="Solution names to analyze (default: quob gurobi).")
    parser.add_argument("--cardinality", type=int, default=300, help="Cardinality used during optimization (default: 300 for Russell 3000).")
    parser.add_argument("--rebalancing", type=int, default=12, help="Rebalancing frequency in months (default: 12, matching the notebook).")
    parser.add_argument("--start_date", default="2014-01-02", help="Training start date used for the portfolios.")
    parser.add_argument("--end_date", default="2023-12-31", help="Training end date used for the portfolios.")
    parser.add_argument("--output_dir", default=None, help="Directory to write plots (default: <result_path>/analysis_<index>_<cardinality>).")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    output_dir = Path(cli_args.output_dir) if cli_args.output_dir else Path(cli_args.result_path) / f"analysis_{cli_args.index}_{cli_args.cardinality}"
    output_dir.mkdir(parents=True, exist_ok=True)

    method_paths = {}
    for solution in cli_args.solutions:
        path = Path(cli_args.result_path) / f"portfolio_{cli_args.index}_{solution}_{cli_args.cardinality}.json"
        method_paths[solution] = path

    rendements: Dict[str, ReturnSeries] = {}
    tracking_errors_all: Dict[str, Dict[pd.Timestamp, float]] = {}
    mae_all: Dict[str, Dict[pd.Timestamp, float]] = {}
    indice_reference: ReturnSeries | None = None

    for method, path in method_paths.items():
        print(f"Traitement : {method} ({path})")
        args = _build_args(cli_args, method)
        rp, ri, te, ae = extract_timeseries(path, args)
        rendements[method] = rp
        tracking_errors_all[method] = te
        mae_all[method] = ae
        if indice_reference is None:
            indice_reference = ri

    if indice_reference is None:
        raise RuntimeError("No portfolios were loaded; nothing to analyze.")

    _plot_cumulative(rendements, indice_reference, output_dir)
    _plot_tracking_errors(tracking_errors_all, output_dir)
    _plot_mae(mae_all, output_dir)
    _plot_combined(rendements, indice_reference, output_dir)
    _plot_error_distributions(rendements, indice_reference, output_dir)

    print(f"✅ Graphiques sauvegardés dans : {output_dir}")


if __name__ == "__main__":
    main()
