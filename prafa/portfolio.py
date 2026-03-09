import numpy as np
from prafa.universe import Universe
from prafa.quob import QUOB
from prafa.matrix_utils import compute_dcor_matrix, compute_simplecor_matrix
from datetime import datetime
import pandas as pd
import pickle
import os


class Portfolio:
    """Orchestrates rebalancing decisions and stores the resulting portfolios.

    Each call to ``rebalance_portfolio()`` delegates to ``Solution`` for
    stock selection and weighting, then records the output keyed by the
    end date of the training window.
    """

    def __init__(self, universe: Universe) -> None:
        self.universe = universe
        self.portfolios: dict = {}

    def rebalance_portfolio(
        self,
        start_datetime: datetime,
        end_datetime: datetime,
    ) -> dict:
        self.universe.new_universe(start_datetime, end_datetime)
        sol = Solution(self)

        weights = sol.solve()
        if isinstance(weights, dict):
            weights = pd.Series(weights)
        elif not isinstance(weights, pd.Series):
            weights = pd.Series(weights, index=self.universe.get_stock_name_in_order())

        audit = getattr(self.universe, "last_cleaning_stats", {})
        self.portfolios[end_datetime] = {
            "weights": weights,
            "calendar_hash": audit.get("calendar_hash"),
            "calendar_count": audit.get("calendar_count"),
        }
        return self.portfolios[end_datetime]

    def get_universe(self) -> Universe:
        return self.universe

    def save_portfolio(self) -> None:
        result_path = self.universe.args.result_path
        os.makedirs(result_path, exist_ok=True)
        suffix = "_phase17_no_strat" if getattr(self.universe.args, "min_trading_frac", 0.50) == 0.0 else ""
        path = (
            f"{result_path}/portfolio_{self.universe.args.index}_"
            f"{self.universe.args.solution_name}_{self.universe.args.cardinality}{suffix}.pkl"
        )
        with open(path, "wb") as f:
            pickle.dump(self.portfolios, f)
        print(f"Portfolio saved to {path}")


class Solution:
    """Builds portfolio weights for a given training window.

    Dispatches to ``quob`` (QP weights, no market-cap data required) or
    ``stratified_quob`` (cap-weighted, requires mktcap_stocks.csv).
    """

    def __init__(self, portfolio: Portfolio) -> None:
        self.portfolio = portfolio
        self.universe = portfolio.get_universe()
        self.solution_name = self.universe.args.solution_name
        self.K = self.universe.args.cardinality

        distance_method = getattr(self.universe.args, "distance_method", "dcor")
        self.simple_corr = distance_method == "pearson"

        self.new_return = self.universe.get_stocks_returns().values
        self.new_index = self.universe.get_index_returns().values
        self.stock_list = self.universe.get_stock_name_in_order()

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def quob(self) -> np.ndarray:
        """Run QUOB with QP weights (no cap-weighting)."""
        obj = QUOB(
            self.new_return,
            self.new_index,
            self.K,
            simple_corr=self.simple_corr,
            replicator_cores=self.universe.args.replicator_cores,
            time_limit=self.universe.args.time_limit,
            d_scale=getattr(self.universe.args, "d_scale", 1.0),
        )
        return obj.get_weights()

    def stratified_quob(self) -> np.ndarray:
        """Single-pool k-medoids with cap-weighting.

        Selects K medoids from all stocks passing training filters via ReplicaTOR,
        then assigns each medoid a weight equal to the sum of market caps in its
        cluster, normalised to 1.

        Pool B (stocks passing ``max_missing_frac`` but failing ``min_trading_frac``)
        are assigned to their nearest medoid by Pearson correlation and contribute
        their market cap to that cluster, even though they are not held.  This
        ensures the cap weights reflect the full index composition rather than
        only the liquid subset.

        Falls back to ``quob()`` if market-cap data is unavailable (e.g. SP500).
        """
        mktcap_weights = getattr(self.universe, "mktcap_weights", None)
        if mktcap_weights is None:
            print("No market-cap weights available; falling back to regular QUOB.")
            return self.quob()

        full_mktcap = getattr(self.universe, "full_mktcap_weights", mktcap_weights)
        pre_liq_columns = list(getattr(self.universe, "pre_liquidity_columns", self.stock_list))
        filtered_stocks = list(self.stock_list)
        filtered_set = set(filtered_stocks)

        if full_mktcap is not None and len(full_mktcap) == len(pre_liq_columns):
            full_mktcap_dict = {s: float(full_mktcap[i]) for i, s in enumerate(pre_liq_columns)}
        else:
            print("Warning: mktcap dimension mismatch; using Pool-A-only cap weights.")
            pool_a_dict = {s: float(mktcap_weights[i]) for i, s in enumerate(filtered_stocks)}
            full_mktcap_dict = {s: pool_a_dict.get(s, 0.0) for s in pre_liq_columns}

        n_total = self.new_return.shape[1]
        compute_dist = compute_simplecor_matrix if self.simple_corr else compute_dcor_matrix
        D_full = compute_dist(self.new_return)
        print(f"QUOB (single pool): n={n_total} stocks, K={self.K}")

        quob_full = QUOB(
            self.new_return,
            self.new_index,
            self.K,
            precomputed_dist=D_full,
            simple_corr=self.simple_corr,
            replicator_cores=self.universe.args.replicator_cores,
            time_limit=self.universe.args.time_limit,
            d_scale=getattr(self.universe.args, "d_scale", 1.0),
        )
        _ = quob_full.get_weights()
        medoid_local = quob_full.idx
        medoid_rets = self.new_return[:, medoid_local]
        K_full = quob_full.K

        # Accumulate cluster market caps via ReplicaTOR assignments (or nearest-medoid fallback)
        cluster_mktcap = np.zeros(K_full)
        if quob_full.cluster_assignments is not None:
            for stock_pos, cluster_pos in enumerate(quob_full.cluster_assignments):
                if 0 <= cluster_pos < K_full:
                    cluster_mktcap[cluster_pos] += full_mktcap_dict.get(filtered_stocks[stock_pos], 0.0)
        else:
            for stock_pos, cluster_pos in enumerate(self._nearest_medoid(self.new_return, medoid_rets)):
                cluster_mktcap[cluster_pos] += full_mktcap_dict.get(filtered_stocks[stock_pos], 0.0)

        # Pool B: illiquid stocks not held but whose cap weight must be attributed
        pool_b = [s for s in pre_liq_columns if s not in filtered_set]
        pool_b_rets, available_b = self._pool_b_returns(pool_b)
        if pool_b_rets is not None and len(available_b) > 0:
            for bi, cluster_pos in enumerate(self._nearest_medoid(pool_b_rets, medoid_rets)):
                cluster_mktcap[cluster_pos] += full_mktcap_dict.get(available_b[bi], 0.0)

        total = cluster_mktcap.sum()
        cluster_mktcap = cluster_mktcap / total if total > 0 else np.ones(K_full) / K_full

        global_weights = np.zeros(n_total)
        for k, med_i in enumerate(medoid_local):
            global_weights[med_i] = cluster_mktcap[k]
        return global_weights

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pool_b_returns(self, pool_b_stocks: list[str]) -> tuple[np.ndarray | None, list[str]]:
        """Return (returns_matrix, stock_names) for Pool B stocks in the training window."""
        available = [s for s in pool_b_stocks if s in self.universe.df_return_all.columns]
        if not available:
            return None, []
        rets = self.universe.df_return_all.loc[self.universe.df_return.index, available].fillna(0.0).values
        hard_clip = getattr(self.universe.args, "hard_clip", 1.0)
        if hard_clip > 0:
            rets = np.clip(rets, -hard_clip, hard_clip)
        return rets, available

    def _nearest_medoid(self, stock_returns: np.ndarray, medoid_returns: np.ndarray) -> np.ndarray:
        """Assign each stock to its nearest medoid by maximum Pearson correlation."""
        def _normalize(X: np.ndarray) -> np.ndarray:
            X = X - X.mean(axis=0)
            norms = np.linalg.norm(X, axis=0)
            return X / np.where(norms > 0, norms, 1.0)

        corr = _normalize(stock_returns).T @ _normalize(medoid_returns)
        return np.argmax(corr, axis=1)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def solve(self) -> np.ndarray:
        if self.solution_name == "quob":
            return self.quob()
        elif self.solution_name == "quob_stratified":
            return self.stratified_quob()
        else:
            raise ValueError(
                f"Unknown solution name: '{self.solution_name}'. "
                "Choose from: quob, quob_stratified."
            )
