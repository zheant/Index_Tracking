"""Analyse des portefeuilles générés par l'optimisation.

Ce module reproduit les visualisations et statistiques exploratoires qui
étaient produites dans ``analyses_resultats.ipynb`` afin de les rendre
reproductibles pour n'importe quel indice (par exemple le Russell 2000).

Utilisation rapide
------------------

.. code-block:: bash

    source .venv/bin/activate
    python scripts/analyze_portfolio.py \
        --index russel2000 \
        --portfolio quob=results/portfolio_russel2000_quob_300.json \
        --start-date 2014-01-02 \
        --end-date 2023-12-31 \
        --output-dir analyses/russel2000

Le script charge le portefeuille (fichier ``pickle`` produit par
``Portfolio.save_portfolio``), reconstruit les rendements hors-échantillon,
calcule les statistiques (tracking error, erreurs absolues, etc.) puis
exporte les figures et les séries temporelles sous ``--output-dir``.
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

try:  # seaborn est optionnel : on bascule automatiquement sur matplotlib.
    import seaborn as sns  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - dépendance optionnelle
    sns = None

from prafa.universe import Universe


@dataclass
class AnalysisResult:
    """Conteneur pour les séries calculées lors du backtest."""

    portfolio_returns: pd.Series
    index_returns: pd.Series
    tracking_error: pd.Series
    absolute_error: pd.Series


def _parse_portfolio_argument(raw_values: Iterable[str]) -> Dict[str, Path]:
    """Convertit les couples ``label=chemin`` fournis sur la CLI."""

    portfolios: Dict[str, Path] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise argparse.ArgumentTypeError(
                "Chaque argument --portfolio doit être de la forme nom=chemin"
            )
        label, path_str = raw.split("=", 1)
        path = Path(path_str).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Fichier portefeuille introuvable : {path}")
        portfolios[label.strip()] = path
    if not portfolios:
        raise argparse.ArgumentTypeError("Au moins un portefeuille est requis")
    return portfolios


def _build_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Construit l'objet ``Namespace`` attendu par :class:`Universe`."""

    namespace = argparse.Namespace()
    namespace.index = args.index
    namespace.data_path = args.data_path
    namespace.result_path = args.result_path
    namespace.solution_name = args.default_solution_name
    namespace.rebalancing = args.rebalancing
    namespace.cardinality = args.cardinality
    namespace.start_date = args.start_date
    namespace.end_date = args.end_date
    return namespace


def _load_portfolios(path: Path) -> Dict[pd.Timestamp, np.ndarray]:
    """Charge la structure picklée produite par :meth:`Portfolio.save_portfolio`."""

    with open(path, "rb") as handle:
        raw = pickle.load(handle)

    portfolios: Dict[pd.Timestamp, np.ndarray] = {}
    for key, weights in raw.items():
        timestamp = pd.Timestamp(key)
        portfolios[timestamp] = np.asarray(weights, dtype=float)
    return dict(sorted(portfolios.items()))


def _compute_out_of_sample(
    universe_args: argparse.Namespace,
    portfolios: Dict[pd.Timestamp, np.ndarray],
    analysis_end: pd.Timestamp,
) -> AnalysisResult:
    """Reproduit la logique d'évaluation hors-échantillon du notebook."""

    universe = Universe(universe_args)

    sorted_dates = list(sorted(portfolios.keys()))
    if len(sorted_dates) < 2:
        raise ValueError(
            "Le portefeuille doit contenir au moins deux dates de rééquilibrage"
        )

    portfolio_returns: list[pd.Series] = []
    index_returns: list[pd.Series] = []
    tracking_error: dict[pd.Timestamp, float] = {}
    absolute_error: dict[pd.Timestamp, float] = {}

    for idx, rebalance_date in enumerate(sorted_dates):
        start = pd.Timestamp(rebalance_date)
        if start > analysis_end:
            break

        if idx + 1 < len(sorted_dates):
            next_date = pd.Timestamp(sorted_dates[idx + 1]) - BDay(1)
            end = min(next_date, analysis_end)
        else:
            end = analysis_end

        if end < start:
            continue

        universe.new_universe(start, end, training=False)
        stock_returns = universe.get_stocks_returns()
        index_slice = universe.get_index_returns()
        weights = portfolios[rebalance_date]

        if stock_returns.shape[1] != weights.shape[0]:
            raise ValueError(
                "Incohérence entre le nombre de titres dans les données et la longueur "
                "du vecteur de poids sauvegardé."
            )

        portfolio_series = pd.Series(
            stock_returns.values @ weights,
            index=stock_returns.index,
            name="portfolio_return",
        )
        index_series = pd.Series(index_slice.values, index=index_slice.index, name="index_return")

        diff = portfolio_series - index_series
        tracking_error[start] = float(diff.std(ddof=0))
        absolute_error[start] = float(diff.abs().mean())

        portfolio_returns.append(portfolio_series)
        index_returns.append(index_series)

    if not portfolio_returns:
        raise ValueError(
            "Impossible de calculer les rendements hors-échantillon : vérifiez les dates fournies."
        )

    portfolio_concat = pd.concat(portfolio_returns).sort_index()
    index_concat = pd.concat(index_returns).sort_index()

    return AnalysisResult(
        portfolio_returns=portfolio_concat,
        index_returns=index_concat,
        tracking_error=pd.Series(tracking_error).sort_index(),
        absolute_error=pd.Series(absolute_error).sort_index(),
    )


def _plot_cumulative_returns(
    results: Dict[str, AnalysisResult],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    index_reference: pd.Series | None = None
    for label, analysis in results.items():
        cumulative = (analysis.portfolio_returns + 1.0).cumprod()
        ax.plot(cumulative.index, cumulative.values, label=f"Portefeuille – {label}")
        if index_reference is None:
            index_reference = (analysis.index_returns + 1.0).cumprod()

    if index_reference is not None:
        ax.plot(index_reference.index, index_reference.values, label="Indice", color="black")

    ax.set_title("Rendements cumulés")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rendement cumulé")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cumulative_returns.png", dpi=300)
    plt.close(fig)


def _plot_absolute_differences(
    results: Dict[str, AnalysisResult],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    reference = None
    for label, analysis in results.items():
        reference = analysis.index_returns
        diff = analysis.portfolio_returns - analysis.index_returns
        ax.plot(diff.index, diff.values, label=f"Écart – {label}", alpha=0.7)

    ax.set_title("Écart de réplication (r_portefeuille - r_indice)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Écart")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "absolute_differences.png", dpi=300)
    plt.close(fig)


def _plot_error_distribution(
    results: Dict[str, AnalysisResult],
    output_dir: Path,
) -> None:
    rows = len(results)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 4 * rows), sharex=True)
    if rows == 1:
        axes = [axes]

    for ax, (label, analysis) in zip(axes, results.items()):
        errors = analysis.portfolio_returns - analysis.index_returns
        if sns is not None:
            sns.histplot(errors, bins=100, kde=True, ax=ax, color="steelblue")
        else:  # pragma: no cover - dépend des bibliothèques installées
            ax.hist(errors, bins=100, color="steelblue", alpha=0.8)
        stats_text = (
            f"Moyenne : {errors.mean():.6f}\n"
            f"Écart-type : {errors.std(ddof=0):.6f}\n"
            f"Skewness : {errors.skew():.3f}\n"
            f"Kurtosis : {errors.kurt():.3f}"
        )
        ax.axvline(errors.mean(), color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"Distribution des erreurs – {label}")
        ax.set_ylabel("Fréquence")
        ax.text(
            0.99,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="top",
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
        )

    axes[-1].set_xlabel("Erreur de réplication")
    fig.tight_layout()
    fig.savefig(output_dir / "error_distribution.png", dpi=300)
    plt.close(fig)


def _export_series(label: str, analysis: AnalysisResult, output_dir: Path) -> None:
    base = output_dir / label
    base.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "portfolio": analysis.portfolio_returns,
            "index": analysis.index_returns,
        }
    ).to_csv(base / "returns.csv")

    analysis.tracking_error.rename("tracking_error").to_csv(base / "tracking_error.csv")
    analysis.absolute_error.rename("absolute_error").to_csv(base / "absolute_error.csv")


def _export_summary(results: Dict[str, AnalysisResult], output_dir: Path) -> Path:
    rows = []
    index_reference: pd.Series | None = None
    for label, analysis in results.items():
        portfolio = analysis.portfolio_returns
        index_returns = analysis.index_returns
        if index_reference is None:
            index_reference = index_returns
        diff = portfolio - index_returns
        rows.append(
            {
                "method": label,
                "cumulative_portfolio_return": float((portfolio + 1.0).prod() - 1.0),
                "cumulative_index_return": float((index_returns + 1.0).prod() - 1.0),
                "tracking_error_mean": float(diff.std(ddof=0)),
                "mean_absolute_error": float(diff.abs().mean()),
                "max_absolute_deviation": float(diff.abs().max()),
            }
        )

    summary = pd.DataFrame(rows).set_index("method")
    summary_path = output_dir / "summary_metrics.csv"
    summary.to_csv(summary_path)
    return summary_path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse des portefeuilles optimisés")
    parser.add_argument(
        "--portfolio",
        dest="portfolio",
        action="append",
        required=True,
        help="Chemins vers les portefeuilles à analyser, sous la forme nom=chemin",
    )
    parser.add_argument("--index", default="russel2000", help="Nom de l'indice analysé")
    parser.add_argument(
        "--start-date",
        default="2014-01-02",
        help="Première date disponible dans les données (format ISO)",
    )
    parser.add_argument(
        "--end-date",
        default="2023-12-31",
        help="Dernière date d'analyse (format ISO)",
    )
    parser.add_argument(
        "--analysis-end-date",
        default=None,
        help="Fin de la période hors-échantillon (format ISO). Par défaut utilise --end-date.",
    )
    parser.add_argument(
        "--cardinality",
        type=int,
        default=300,
        help="Cardinalité utilisée lors de l'optimisation (uniquement pour les métadonnées)",
    )
    parser.add_argument(
        "--rebalancing",
        type=int,
        default=12,
        help="Fréquence de rééquilibrage en mois (métadonnée)",
    )
    parser.add_argument(
        "--data-path",
        default="financial_data",
        help="Répertoire contenant les données de marché",
    )
    parser.add_argument(
        "--result-path",
        default="results",
        help="Répertoire où les portefeuilles sont sauvegardés",
    )
    parser.add_argument(
        "--default-solution-name",
        default="quob",
        help="Nom de la solution utilisé pour initialiser Universe",
    )
    parser.add_argument(
        "--output-dir",
        default="analyses",
        help="Répertoire de sortie pour les figures et tableaux",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    portfolios = _parse_portfolio_argument(args.portfolio)

    universe_args = _build_namespace(args)
    analysis_end = pd.Timestamp(args.analysis_end_date or args.end_date)

    results: Dict[str, AnalysisResult] = {}
    for label, path in portfolios.items():
        portfolio_data = _load_portfolios(path)
        results[label] = _compute_out_of_sample(universe_args, portfolio_data, analysis_end)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, analysis in results.items():
        _export_series(label, analysis, output_dir)

    summary_path = _export_summary(results, output_dir)
    _plot_cumulative_returns(results, output_dir)
    _plot_absolute_differences(results, output_dir)
    _plot_error_distribution(results, output_dir)

    print(f"Analyse terminée. Résumé sauvegardé dans {summary_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
