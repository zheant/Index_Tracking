from __future__ import annotations

import pickle
from datetime import datetime
import time
from typing import Dict, TYPE_CHECKING

import numpy as np

from prafa.quob import QUOB
from prafa.universe import Universe

if TYPE_CHECKING:  # pragma: no cover - optional dependency only needed for type checking
    from prafa.gurobi import Gurobi


class Portfolio:
    """Handle portfolio rebalancing across time windows."""

    def __init__(self, universe: Universe) -> None:
        self.universe = universe
        # Mapping from rebalancing date to the weights produced by the solver.
        self.portfolios: Dict[datetime, np.ndarray] = {}

    def rebalance_portfolio(self, start_datetime: datetime, end_datetime: datetime) -> np.ndarray:
        """Recompute the portfolio weights for the requested time window."""
        self.universe.new_universe(start_datetime, end_datetime)
        solution = Solution(self, start_datetime, end_datetime)
        start_ts = time.perf_counter()
        self.portfolios[end_datetime] = solution.solve()
        elapsed = time.perf_counter() - start_ts
        print(
            "✅ Fenêtre optimisée :",
            f"{start_datetime.date()} → {end_datetime.date()}",
            f"(durée {elapsed / 3600:.2f} h)",
        )
        return self.portfolios[end_datetime]

    def get_universe(self) -> Universe:
        return self.universe

    def save_portfolio(self) -> None:
        path = (
            f"{self.universe.args.result_path}/"
            f"portfolio_{self.universe.args.index}_{self.universe.args.solution_name}_{self.universe.args.cardinality}.json"
        )
        with open(path, "wb") as handle:
            pickle.dump(self.portfolios, handle)
            print("les portefeuilles ont été enregistrés!! pret a réalisé le backtest")

    def __del__(self) -> None:  # pragma: no cover - destructor semantics are hard to unit test.
        if self.portfolios:
            self.save_portfolio()


class Solution:
    """Wrapper around the optimisation solvers supported by the project."""

    def __init__(self, portfolio: Portfolio, start_datetime: datetime, end_datetime: datetime) -> None:
        self.portfolio = portfolio
        self.universe = portfolio.get_universe()
        self.solution_name = self.universe.args.solution_name
        self.cardinality = self.universe.args.cardinality
        self.start_datetime = start_datetime
        self.end_datetime = end_datetime

        # numpy arrays expected by the solvers
        self.stocks_returns = self.universe.get_stocks_returns().values
        self.index_returns = self.universe.get_index_returns().values

    def _solve_with_quob(self, simple_corr: bool = False) -> np.ndarray:
        solver = QUOB(
            self.stocks_returns,
            self.index_returns,
            self.cardinality,
            simple_corr=simple_corr,
            replicator_bin=getattr(self.universe.args, "replicator_bin", None),
            replicator_time_limit=getattr(self.universe.args, "replicator_time_limit", 300.0),
            replicator_cores=getattr(self.universe.args, "replicator_cores", 1),
        )
        return solver.get_weights()

    def _solve_with_gurobi(self, simple_corr: bool = False) -> np.ndarray:
        try:
            from prafa.gurobi import Gurobi
        except ModuleNotFoundError as exc:  # pragma: no cover - hard to simulate without uninstalling gurobi
            raise ModuleNotFoundError(
                "La solution 'gurobi' requiert l'installation du paquet 'gurobipy'. "
                "Veuillez installer Gurobi avant d'utiliser cette option."
            ) from exc

        window_label = f"{self.start_datetime.date()}_{self.end_datetime.date()}"
        solver = Gurobi(
            self.stocks_returns,
            self.index_returns,
            self.cardinality,
            simple_corr=simple_corr,
            time_limit=getattr(self.universe.args, "gurobi_time_limit", 10800.0),
            log_dir=getattr(self.universe.args, "gurobi_log_dir", None),
            log_label=window_label,
        )
        return solver.get_weights()

    def solve(self) -> np.ndarray:
        if self.solution_name == "quob":
            return self._solve_with_quob(simple_corr=False)
        if self.solution_name == "quob_cor":
            return self._solve_with_quob(simple_corr=True)
        if self.solution_name == "gurobi":
            return self._solve_with_gurobi(simple_corr=False)
        if self.solution_name == "gurobi_cor":
            return self._solve_with_gurobi(simple_corr=True)

        raise ValueError(f"Solution '{self.solution_name}' non supportée")
