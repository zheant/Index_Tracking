import numpy as np
from prafa.universe import Universe
from prafa.quob import QUOB
from prafa.matrix_utils import compute_dcor_matrix, compute_simplecor_matrix
from prafa.gurobi import Gurobi
from datetime import datetime
import pandas as pd
import pickle
import os
from scipy.optimize import minimize

"""
Cette classe centralise le code. Elle qui mets a jour l'univers et ensuite calcule L'optimisation et stack la reponse ici avec la date
Faire bien attention de bien ajuster universe avant optimization parce l'universe est réutiliser dans L'optimisation , mais on la récupere
a partir de portfolio
"""


class Portfolio:
    def __init__(self, universe: Universe):
        self.universe = universe

        self.portfolios = {}  # Dictionnaire pour stocker les portefeuilles par date (le portfeuille est un dictionnaire de poids)

    def rebalance_portfolio(self,
        start_datetime : datetime,
        end_datetime : datetime
    )   :
        #la fenetre de temps est celle de l'entrainement donc, on regarde composition de la end_date et se sert des
        #données passées pour résoudre le probleme d'optimisation et ainsi trouver les poids optimiaux
        self.universe.new_universe(start_datetime, end_datetime)
        sol = Solution(self)

        weights = sol.solve()
        if isinstance(weights, dict):
            weights = pd.Series(weights)
        elif not isinstance(weights, pd.Series):
            weights = pd.Series(weights, index=self.universe.get_stock_namme_in_order())

        audit = getattr(self.universe, "last_cleaning_stats", {})
        self.portfolios[end_datetime] = {
            "weights": weights,
            "calendar_hash": audit.get("calendar_hash"),
            "calendar_count": audit.get("calendar_count"),
        }
        return self.portfolios[end_datetime] # dictionnire contenant poids


    def get_universe(self) -> Universe:
        return self.universe

    def save_portfolio(self):
        result_path = self.universe.args.result_path
        os.makedirs(result_path, exist_ok=True)
        suffix = ""
        if getattr(self.universe.args, 'exclude_pool_b_capweight', False):
            suffix = "_phase16_pool_a_only"
        elif getattr(self.universe.args, 'phase18_qp_index', False):
            suffix = "_phase18_qp_index"
        elif getattr(self.universe.args, 'min_trading_frac', 0.50) == 0.0 and getattr(self.universe.args, 'no_stratification', False):
            suffix = "_phase17_no_strat"
        elif getattr(self.universe.args, 'min_trading_frac', 0.50) == 0.0:
            suffix = "_phase17_no_liq_filter"
        path = f'{result_path}/portfolio_{self.universe.args.index}_{self.universe.args.solution_name}_{self.universe.args.cardinality}{suffix}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.portfolios, f)
            print('les portefeuilles ont été enregistrés!! pret a réalisé le backtest')




"""
La classe solution est la classe permettant de résoudre le problème d'optimisation.
"""


class Solution:

    def __init__(
        self,
        portfolio : Portfolio,
        ):

        self.portfolio = portfolio
        self.universe =  portfolio.get_universe()
        self.solution_name = self.universe.args.solution_name
        self.K = self.universe.args.cardinality
        self.distance_method = getattr(self.universe.args, "distance_method", "dcor")
        self.simple_corr = self.distance_method == "pearson"

        # Preserve explicit *_cor solution names while allowing CLI flag override.
        if self.solution_name.endswith("_cor"):
            self.simple_corr = True

        self.new_return = self.universe.get_stocks_returns().values
        self.new_index = self.universe.get_index_returns().values
        self.stock_list = self.universe.get_stock_namme_in_order()




    def quob(self):
        obj = QUOB(
            self.new_return,
            self.new_index,
            self.universe.args.cardinality,
            simple_corr=self.simple_corr,
            replicator_cores=self.universe.args.replicator_cores,
            time_limit=self.universe.args.time_limit,
            index_weights=getattr(self.universe, 'mktcap_weights', None),
            d_scale=getattr(self.universe.args, 'd_scale', 1.0),
        )
        return obj.get_weights()

    def stratified_quob(self):
        """Two-pool stratified k-medoids with Neyman allocation.

        Pool A (selection pool): stocks that passed all training filters. Used for
        distance matrix computation and ReplicaTOR medoid selection.

        Pool B (weighting pool): stocks that passed the NaN filter but failed the
        liquidity filter. Too illiquid to be reliable medoids, but their market-cap
        weight must be correctly attributed to the nearest medoid to mirror the index
        construction — eliminating the quality/survivorship bias from renormalising
        cap weights over the filtered subset only.

        Strata boundaries are defined on the full pre-liquidity universe (Pool A + B),
        sorted by market cap. After QUOB selects medoids from Pool A per stratum,
        each Pool B stock is assigned to its nearest medoid (max Pearson correlation)
        within the same stratum. Cluster weights aggregate ALL stocks.

        Falls back to regular QUOB if market-cap data is unavailable.
        """
        mktcap_weights = getattr(self.universe, 'mktcap_weights', None)
        if mktcap_weights is None:
            print("No market-cap weights available; falling back to regular QUOB.")
            return self.quob()

        # Two-pool data
        full_mktcap = getattr(self.universe, 'full_mktcap_weights', mktcap_weights)
        pre_liq_columns = list(getattr(self.universe, 'pre_liquidity_columns', self.stock_list))
        filtered_stocks = list(self.stock_list)  # Pool A stock names
        filtered_set = set(filtered_stocks)

        # Cap-weight dict for all pre-liquidity stocks
        full_mktcap_dict = {s: float(full_mktcap[i]) for i, s in enumerate(pre_liq_columns)}

        # --- 1. Stratum boundaries on FULL pre-liquidity universe ---
        strata_large_size = getattr(self.universe.args, 'strata_large_size', 1000)
        sorted_pre_liq = np.argsort(full_mktcap)[::-1]
        n_large_full = max(1, min(strata_large_size, len(sorted_pre_liq) - 1))

        large_pre_liq_set = {pre_liq_columns[i] for i in sorted_pre_liq[:n_large_full]}
        small_pre_liq_set = {pre_liq_columns[i] for i in sorted_pre_liq[n_large_full:]}

        # Pool A indices (into self.new_return) and stock names per stratum
        pool_a_large_local = [i for i, s in enumerate(filtered_stocks) if s in large_pre_liq_set]
        pool_a_small_local = [i for i, s in enumerate(filtered_stocks) if s in small_pre_liq_set]
        pool_a_large_stocks = [filtered_stocks[i] for i in pool_a_large_local]
        pool_a_small_stocks = [filtered_stocks[i] for i in pool_a_small_local]

        # Pool B stock names per stratum
        pool_b_large_stocks = [s for s in pre_liq_columns if s in large_pre_liq_set and s not in filtered_set]
        pool_b_small_stocks = [s for s in pre_liq_columns if s in small_pre_liq_set and s not in filtered_set]

        if len(pool_a_large_local) == 0 or len(pool_a_small_local) == 0:
            print("Warning: one stratum has no Pool A stocks; falling back to regular QUOB.")
            return self.quob()

        # Stratum market-cap fractions
        pool_b_in_capweight = not getattr(self.universe.args, 'exclude_pool_b_capweight', False)
        if pool_b_in_capweight:
            # Full pre-liquidity universe (Pool A + Pool B) — Phase 12 reference
            w_large = sum(full_mktcap_dict.get(s, 0.0) for s in large_pre_liq_set)
            w_small = sum(full_mktcap_dict.get(s, 0.0) for s in small_pre_liq_set)
        else:
            # Pool A only — Phase 16: eliminates liquidity bias
            w_large = sum(full_mktcap_dict.get(s, 0.0) for s in pool_a_large_stocks)
            w_small = sum(full_mktcap_dict.get(s, 0.0) for s in pool_a_small_stocks)
        total_w = w_large + w_small
        w_large = w_large / total_w if total_w > 0 else 0.5
        w_small = w_small / total_w if total_w > 0 else 0.5

        # --- 2. Distance matrices for Pool A (reused for Neyman + QUOB) ---
        compute_dist = compute_simplecor_matrix if self.simple_corr else compute_dcor_matrix
        returns_large = self.new_return[:, pool_a_large_local]
        returns_small = self.new_return[:, pool_a_small_local]
        n_l, n_s = len(pool_a_large_local), len(pool_a_small_local)

        D_large = compute_dist(returns_large)
        D_small = compute_dist(returns_small)

        d_bar_large = D_large.sum() / (n_l * (n_l - 1)) if n_l > 1 else 0.0
        d_bar_small = D_small.sum() / (n_s * (n_s - 1)) if n_s > 1 else 0.0

        # --- 3. Neyman allocation: K_h ∝ n_h × d̄_h ---
        score_large = n_l * d_bar_large
        score_small = n_s * d_bar_small
        total_score = score_large + score_small
        K_large = max(1, round(self.K * score_large / total_score)) if total_score > 0 else max(1, round(self.K * n_l / (n_l + n_s)))
        K_small = self.K - K_large
        K_large = min(K_large, n_l)
        K_small = min(max(K_small, 1), n_s)
        K_large = self.K - K_small

        print(
            f"Stratified QUOB (Neyman, two-pool): "
            f"large (A:{n_l}+B:{len(pool_b_large_stocks)}, {w_large:.1%} cap, d̄={d_bar_large:.4f}) → K={K_large} | "
            f"small (A:{n_s}+B:{len(pool_b_small_stocks)}, {w_small:.1%} cap, d̄={d_bar_small:.4f}) → K={K_small}"
        )

        common_kwargs = dict(
            simple_corr=self.simple_corr,
            replicator_cores=self.universe.args.replicator_cores,
            time_limit=self.universe.args.time_limit,
            d_scale=getattr(self.universe.args, 'd_scale', 1.0),
        )

        # --- Phase 17c: no stratification — single QUOB on full Pool A ---
        if getattr(self.universe.args, 'no_stratification', False):
            n_total = self.new_return.shape[1]
            compute_dist = compute_simplecor_matrix if self.simple_corr else compute_dcor_matrix
            D_full = compute_dist(self.new_return)
            quob_full = QUOB(self.new_return, self.new_index, self.K,
                             precomputed_dist=D_full, **common_kwargs)
            _ = quob_full.get_weights()
            medoid_local = quob_full.idx
            medoid_rets = self.new_return[:, medoid_local]
            K_full = quob_full.K
            cluster_mktcap = np.zeros(K_full)
            if quob_full.cluster_assignments is not None:
                for stock_pos, cluster_pos in enumerate(quob_full.cluster_assignments):
                    if 0 <= cluster_pos < K_full:
                        cluster_mktcap[cluster_pos] += full_mktcap_dict.get(filtered_stocks[stock_pos], 0.0)
            else:
                assignments = self._nearest_medoid(self.new_return, medoid_rets)
                for stock_pos, cluster_pos in enumerate(assignments):
                    cluster_mktcap[cluster_pos] += full_mktcap_dict.get(filtered_stocks[stock_pos], 0.0)
            if pool_b_in_capweight:
                pool_b_all = [s for s in pre_liq_columns if s not in filtered_set]
                pool_b_rets, available_b = self._get_pool_b_returns(pool_b_all)
                if pool_b_rets is not None and len(available_b) > 0:
                    assignments_b = self._nearest_medoid(pool_b_rets, medoid_rets)
                    for bi, stock_name in enumerate(available_b):
                        cluster_mktcap[assignments_b[bi]] += full_mktcap_dict.get(stock_name, 0.0)
            total_cluster = cluster_mktcap.sum()
            if total_cluster > 0:
                cluster_mktcap /= total_cluster
            else:
                cluster_mktcap = np.ones(K_full) / K_full
            global_weights = np.zeros(n_total)
            for k, med_i in enumerate(medoid_local):
                global_weights[med_i] = cluster_mktcap[k]
            return global_weights

        # --- 4. Run QUOB on Pool A per stratum (medoid selection only) ---
        # get_weights() is called for its side-effect: populates .idx and .cluster_assignments.
        # The QP weights returned are discarded; cap-weighting is computed below.
        quob_large = QUOB(returns_large, self.new_index, K_large,
                          precomputed_dist=D_large, **common_kwargs)
        _ = quob_large.get_weights()  # triggers stock_picking() → sets .idx and .cluster_assignments

        quob_small = QUOB(returns_small, self.new_index, K_small,
                          precomputed_dist=D_small, **common_kwargs)
        _ = quob_small.get_weights()

        # --- 5. Weight assignment ---
        n_total = self.new_return.shape[1]
        global_weights = np.zeros(n_total)

        # Phase 18: QP targeting r_index directly.
        # Collect medoid global indices from both strata, then solve:
        #   min ||R_medoids @ w - r_index||²  s.t. Σw=1, w≥0
        # Pool B is not held (realistic) but its influence passes through the index target.
        if getattr(self.universe.args, 'phase18_qp_index', False):
            medoid_globals = []
            for quob_obj, pool_a_local in [
                (quob_large, pool_a_large_local),
                (quob_small, pool_a_small_local),
            ]:
                for med_local_i in quob_obj.idx:
                    medoid_globals.append(pool_a_local[med_local_i])
            R_med = self.new_return[:, medoid_globals]
            n_med = len(medoid_globals)
            w0 = np.ones(n_med) / n_med
            result = minimize(
                lambda w: np.sum((R_med @ w - self.new_index) ** 2),
                w0,
                method='SLSQP',
                constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1},
                bounds=[(0, 1)] * n_med,
            )
            if not result.success:
                print(f"Warning: Phase 18 QP did not converge: {result.message}")
            for k, gidx in enumerate(medoid_globals):
                global_weights[gidx] = max(0.0, result.x[k])
            return global_weights

        # Cap-weighting (Phase 12 / 16): each medoid receives the sum of mktcap of
        # all stocks in its cluster (Pool A via ReplicaTOR + optionally Pool B).

        for quob_obj, pool_a_local, pool_a_stocks, pool_b_stocks, returns_a, w_stratum in [
            (quob_large, pool_a_large_local, pool_a_large_stocks, pool_b_large_stocks, returns_large, w_large),
            (quob_small, pool_a_small_local, pool_a_small_stocks, pool_b_small_stocks, returns_small, w_small),
        ]:
            K_h = quob_obj.K
            medoid_local = quob_obj.idx  # K_h indices within Pool A stratum
            medoid_rets = returns_a[:, medoid_local]  # T × K_h

            # Pool A contributions via ReplicaTOR cluster assignments
            cluster_mktcap = np.zeros(K_h)
            if quob_obj.cluster_assignments is not None:
                for stock_pos, cluster_pos in enumerate(quob_obj.cluster_assignments):
                    if 0 <= cluster_pos < K_h:
                        cluster_mktcap[cluster_pos] += full_mktcap_dict.get(pool_a_stocks[stock_pos], 0.0)
            else:
                assignments_a = self._nearest_medoid(returns_a, medoid_rets)
                for stock_pos, cluster_pos in enumerate(assignments_a):
                    cluster_mktcap[cluster_pos] += full_mktcap_dict.get(pool_a_stocks[stock_pos], 0.0)

            # Pool B contributions — assign to nearest medoid by Pearson correlation
            # Skipped in Phase 16 (exclude_pool_b_capweight=True): Pool A only cap-weighting
            if pool_b_in_capweight:
                pool_b_rets, available_b = self._get_pool_b_returns(pool_b_stocks)
                if pool_b_rets is not None and len(available_b) > 0:
                    assignments_b = self._nearest_medoid(pool_b_rets, medoid_rets)
                    for bi, stock_name in enumerate(available_b):
                        cluster_mktcap[assignments_b[bi]] += full_mktcap_dict.get(stock_name, 0.0)

            # Normalize within stratum and scale by stratum market-cap fraction
            total_cluster = cluster_mktcap.sum()
            if total_cluster > 0:
                cluster_mktcap /= total_cluster
            else:
                cluster_mktcap = np.ones(K_h) / K_h

            for k, med_local_i in enumerate(medoid_local):
                global_weights[pool_a_local[med_local_i]] += cluster_mktcap[k] * w_stratum

        return global_weights

    def _get_pool_b_returns(self, pool_b_stocks):
        """Return (returns_matrix, available_stocks) for Pool B stocks over the training window.

        Applies fillna(0) and hard_clip — same transforms as Pool A — so that Pearson
        correlation estimates are consistent between pools.
        """
        training_dates = self.universe.df_return.index
        available = [s for s in pool_b_stocks if s in self.universe.df_return_all.columns]
        if not available:
            return None, []
        rets = self.universe.df_return_all.loc[training_dates, available].fillna(0.0).values
        hard_clip = getattr(self.universe.args, 'hard_clip', 1.0)
        if hard_clip > 0:
            rets = np.clip(rets, -hard_clip, hard_clip)
        return rets, available

    def _nearest_medoid(self, stock_returns, medoid_returns):
        """Assign each stock to its nearest medoid by maximum Pearson correlation.

        Vectorized: normalize columns to zero mean and unit norm, then compute
        the full correlation matrix as a single matrix multiply. Falls back to
        medoid 0 for constant series (norm == 0).
        """
        def normalize_cols(X):
            X = X - X.mean(axis=0)
            norms = np.linalg.norm(X, axis=0)
            norms = np.where(norms > 0, norms, 1.0)
            return X / norms

        pb = normalize_cols(stock_returns)   # T × n_stocks
        pm = normalize_cols(medoid_returns)  # T × K_h
        corr_matrix = pb.T @ pm              # n_stocks × K_h
        return np.argmax(corr_matrix, axis=1)

    def gurobi(self):
        obj = Gurobi(
            self.new_return,
            self.new_index,
            self.universe.args.cardinality,
            simple_corr=self.simple_corr,
            time_limit=self.universe.args.time_limit,
        )
        weights = obj.get_weights()
        if weights is None:
            status = getattr(obj, "status", None)
            runtime = getattr(obj, "runtime", None)
            selected = getattr(obj, "selection_cardinality", None)
            print(
                "Warning: Gurobi did not return any weights (status=%s, runtime=%s, "
                "selected=%s); using zero portfolio." % (status, runtime, selected)
            )
            return np.zeros(self.new_return.shape[1])
        return weights

    def gurobi_cor(self):
        obj = Gurobi(
            self.new_return,
            self.new_index,
            self.universe.args.cardinality,
            simple_corr=True,
            time_limit=self.universe.args.time_limit,
        )
        weights = obj.get_weights()
        if weights is None:
            status = getattr(obj, "status", None)
            runtime = getattr(obj, "runtime", None)
            selected = getattr(obj, "selection_cardinality", None)
            print(
                "Warning: Gurobi (correlation) did not return any weights (status=%s, runtime=%s, "
                "selected=%s); using zero portfolio." % (status, runtime, selected)
            )
            return np.zeros(self.new_return.shape[1])
        return weights

    def solve(
        self,
    ) -> dict :
        solution_name = self.solution_name

        if solution_name == 'quob':
            weights = self.quob()
        elif solution_name == 'quob_stratified':
            weights = self.stratified_quob()
        elif solution_name == 'gurobi':
            weights = self.gurobi()
        elif solution_name == 'gurobi_cor':
            weights = self.gurobi_cor()
        else:
            raise ValueError(
                f"Unknown solution name: '{solution_name}'. "
                "Choose from: quob, quob_stratified, gurobi, gurobi_cor."
            )

        return weights
