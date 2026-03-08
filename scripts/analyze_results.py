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
            raise ValueError(
                "Weights are an unlabeled array and length does not match the universe columns: "
                f"{len(series)} != {len(target_cols)}. Regenerate portfolios with labeled weights "
                "or ensure the saved weights align to the cleaned universe."
            )
        series = pd.Series(series.values, index=target_cols)
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

    rp_parts: list[pd.Series] = []
    ri_parts: list[pd.Series] = []
    tracking_errors: Dict[pd.Timestamp, float] = {}
    mae: Dict[pd.Timestamp, float] = {}

    for i in range(n - 1):
        # portfolio_key : date à laquelle le portefeuille a été construit (= fin training)
        portfolio_key = pd.Timestamp(dates[i])
        # Le test commence le JOUR SUIVANT (la dernière observation du training
        # ne doit pas être réévaluée en out-of-sample).
        test_start = portfolio_key + pd.tseries.offsets.BDay(1)
        end_date = pd.Timestamp(dates[i + 1]) - pd.tseries.offsets.BDay(1)

        portfolio_entry = portfolios[portfolio_key]
        if isinstance(portfolio_entry, dict) and "weights" in portfolio_entry:
            weights = portfolio_entry["weights"]
            saved_calendar_hash = portfolio_entry.get("calendar_hash")
            saved_calendar_count = portfolio_entry.get("calendar_count")
        else:
            weights = portfolio_entry
            saved_calendar_hash = None
            saved_calendar_count = None

        universe.new_universe(test_start, end_date, training=False)
        X_test = universe.get_stocks_returns()
        Y_test = universe.get_index_returns()
        weights_series = _align_weights(weights, X_test.columns, target_cardinality=args.cardinality)
        # état juste après alignement ---
        k_target = getattr(args, "cardinality", None)
        x_cols = X_test.shape[1]
        nz_align = int((weights_series.abs() > 1e-12).sum())

        print(
            f"[{test_start.date()}→{end_date.date()}] "
            f"X_cols={x_cols}, nonzero_after_align={nz_align}"
            + (f", K_target={k_target}" if k_target is not None else "")
        )
        if X_test.shape[1] < args.cardinality:
            print(
                f"⚠️ Cleaned universe size ({X_test.shape[1]}) below target cardinality ({args.cardinality}) for window starting {test_start.date()}."
            )

        if not (X_test.index == Y_test.index).all():
            raise ValueError(
                f"Les index de X_test et Y_test ne sont pas alignés pour la fenêtre "
                f"{test_start.date()} → {end_date.date()} !"
            )

        # Les NaN de rendement en test sont traités comme position cash (rendement 0).
        # Le filtre min_presence a été supprimé : il calculait le taux de présence
        # sur TOUTE la fenêtre de test (données futures), introduisant un regard
        # dans le futur. Le fillna(0) ci-dessous gère correctement les délistings
        # et les jours sans cotation — pas besoin de pré-filtrer.
        invested_cols = weights_series[weights_series != 0].index.tolist()
        if not invested_cols:
            print(
                f"⚠️ Window {test_start.date()} → {end_date.date()}: "
                "no non-zero weights after alignment; portfolio is all cash."
            )

        nz_after_align = int((weights_series.abs() > 1e-12).sum())
        print(
            f"[{test_start.date()}→{end_date.date()}] "
            f"nonzero_after_align={nz_after_align}"
        )

        return_outsample = X_test.fillna(0.0).mul(weights_series, axis=1).sum(axis=1)

        # qualité de la série de rendement ---
        nan_rp = int(return_outsample.isna().sum())
        print(
            f"[{test_start.date()}→{end_date.date()}] "
            f"rp_nan_days={nan_rp} / {len(return_outsample)}"
        )

        diff = (return_outsample - Y_test).dropna()
        tracking_errors[X_test.index[-1]] = diff.std()
        mae[X_test.index[-1]] = diff.abs().mean()

        rp_parts.append(return_outsample)
        ri_parts.append(Y_test)

    rendements_portefeuille = pd.concat(rp_parts)
    rendements_indice = pd.concat(ri_parts)
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
    plt.ylabel("Erreur absolue moyenne (MAE)")
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
        skewness = skew(erreurs, nan_policy='omit')
        kurt = kurtosis(erreurs, nan_policy='omit')

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
        missing_policy=cli_args.missing_policy,
        min_presence=getattr(cli_args, "min_presence", 0.90),
        # Paramètres de nettoyage : doivent correspondre exactement à ceux
        # utilisés lors de la génération des portefeuilles (main.py).
        reconstitution_month=getattr(cli_args, "reconstitution_month", 7),
        max_missing_frac=getattr(cli_args, "max_missing_frac", 0.10),
        min_trading_frac=getattr(cli_args, "min_trading_frac", 0.50),
        winsor_sigma=getattr(cli_args, "winsor_sigma", 3.0),
        hard_clip=getattr(cli_args, "hard_clip", 1.0),
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
    parser.add_argument("--min_presence", type=float, default=0.90,
        help="Minimum fraction of non-missing observations required to keep an invested asset in a backtest window (default: 0.90).")
    parser.add_argument("--reconstitution_month", type=int, default=7,
        help="Premier mois où la nouvelle composition est active (défaut 7). Doit correspondre à main.py.")
    parser.add_argument("--max_missing_frac", type=float, default=0.10,
        help="Fraction maximale de NaN tolérée par stock (défaut 0.10). Doit correspondre à main.py.")
    parser.add_argument("--min_trading_frac", type=float, default=0.50,
        help="Fraction minimale de jours à rendement non nul (défaut 0.50). Doit correspondre à main.py.")
    parser.add_argument("--winsor_sigma", type=float, default=3.0,
        help="Seuil de winsorisation en σ (0 pour désactiver, défaut 3.0). Doit correspondre à main.py.")
    parser.add_argument("--hard_clip", type=float, default=1.0,
        help="Clip absolu des rendements aberrants en ±fraction (défaut 1.0 = ±100%%). Doit correspondre à main.py.")
    parser.add_argument(
        "--missing_policy",
        choices=["auto", "strict", "legacy"],
        default="auto",
        help=(
            "Missing-data handling to match portfolio generation: auto chooses legacy for SP500 and strict otherwise; "
            "override to force a specific policy."
        ),
    )
    parser.add_argument("--pkl_path", default=None,
        help="Chemin exact vers un PKL spécifique (override la construction automatique du nom).")
    parser.add_argument("--plots", nargs="+",
        choices=["cumulative", "tracking_errors", "mae", "combined", "distributions"],
        default=["cumulative", "tracking_errors", "mae", "combined", "distributions"],
        help="Graphiques à générer (défaut : tous).")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    if cli_args.missing_policy == "auto":
        cli_args.missing_policy = "legacy" if cli_args.index.lower() == "sp500" else "strict"

    output_dir = Path(cli_args.output_dir) if cli_args.output_dir else Path(cli_args.result_path) / f"analysis_{cli_args.index}_{cli_args.cardinality}"
    output_dir.mkdir(parents=True, exist_ok=True)

    method_paths = {}
    for solution in cli_args.solutions:
        if cli_args.pkl_path:
            path = Path(cli_args.pkl_path)
        else:
            path = Path(cli_args.result_path) / f"portfolio_{cli_args.index}_{solution}_{cli_args.cardinality}.pkl"
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

    plots = set(cli_args.plots)
    if "cumulative" in plots:
        _plot_cumulative(rendements, indice_reference, output_dir)
    if "tracking_errors" in plots:
        _plot_tracking_errors(tracking_errors_all, output_dir)
    if "mae" in plots:
        _plot_mae(mae_all, output_dir)
    if "combined" in plots:
        _plot_combined(rendements, indice_reference, output_dir)
    if "distributions" in plots:
        _plot_error_distributions(rendements, indice_reference, output_dir)

    print(f"✅ Graphiques sauvegardés dans : {output_dir}")


if __name__ == "__main__":
    main()
