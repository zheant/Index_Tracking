import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import subprocess
import re

from prafa.matrix_utils import compute_dcor_matrix, compute_simplecor_matrix


# Default to the compiled ReplicaTOR binary under ~/or_tool/ReplicaTOR/cmake-build unless overridden.
DEFAULT_REPLICATOR_PATH = Path(
    os.environ.get("REPLICATOR_PATH", Path.home() / "or_tool" / "ReplicaTOR" / "cmake-build")
)
DEFAULT_REPLICATOR_CORES = int(os.environ.get("REPLICATOR_CORES", 8))
DEFAULT_TIME_LIMIT = 300.0


class QUOB:
    def __init__(self, stocks_returns, index_returns, K, simple_corr=False,
                 replicator_cores=None, time_limit=None, index_weights=None, d_scale=1.0,
                 precomputed_dist=None):
        #matrice et vecteur numpy
        self.stocks_returns = stocks_returns
        self.index_returns = index_returns
        self.K = K #cardinalité!!
        self.idx = None #liste d'indice des stonks choisit
        self.cluster_assignments = None  # liste d'indice de cluster pour chaque stock (0..K-1)
        self.index_weights = index_weights  # poids market-cap normalisés (len = nb stocks dans l'univers)
        self.d_scale = d_scale  # D_scale_factor pour ReplicaTOR
        self.replicator_cores = replicator_cores or DEFAULT_REPLICATOR_CORES
        self.time_limit = DEFAULT_TIME_LIMIT if time_limit is None else time_limit

        # Each instance gets its own isolated temp directory — safe for parallel runs.
        self._dist_dir = Path(tempfile.mkdtemp(prefix="quob_"))

        # Write distance matrix to disk — use pre-computed matrix if provided (avoids
        # recomputation when the caller already computed it for e.g. Neyman allocation).
        if precomputed_dist is not None:
            n = precomputed_dist.shape[0]
            np.savetxt(self._dist_dir / "dist_matrix.d", precomputed_dist)
            adj = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
            np.savetxt(self._dist_dir / "dist_matrix.adj", adj, fmt="%d")
        elif simple_corr:
            self.matrix_simplecor()
        else:
            self.matrix_dcor()

    def _cleanup(self):
        """Remove the per-instance temp directory after the solve is complete."""
        if self._dist_dir.exists():
            shutil.rmtree(self._dist_dir, ignore_errors=True)

    def matrix_dcor(self):
        mat = compute_dcor_matrix(self.stocks_returns)
        n = mat.shape[0]
        np.savetxt(self._dist_dir / "dist_matrix.d", mat)
        adj = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
        np.savetxt(self._dist_dir / "dist_matrix.adj", adj, fmt="%d")

    def matrix_simplecor(self):
        mat = compute_simplecor_matrix(self.stocks_returns)
        n = mat.shape[0]
        np.savetxt(self._dist_dir / "dist_matrix.d", mat)
        adj = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
        np.savetxt(self._dist_dir / "dist_matrix.adj", adj, fmt="%d")


    def stock_picking(self):
        #résolution du probleme d'optimisation
        #retourne (medoids, cluster_assignments)
        n = self.stocks_returns.shape[1]
        b_scale = 0.5 * (self.K + 1) / n
        param = f"""num_vars {n} #INT number of variables/nodes
                num_k {self.K} #INT number of medoids/exemplars
                B_scale_factor {b_scale} #FLOAT32 scaling factor for model bias, set to 0.5*(num_k+1)/num_vars
                D_scale_factor {self.d_scale} #FLOAT32 scaling factor for model distances, leave at 1
                problem_path {self._dist_dir}/
                problem_name dist_matrix
                cost_answer -1000000 #FLOAT32 target cost to allow program to exit early if found, set to large neg value if you don't want an early exit
                T_max 0.01 #FLOAT32 parallel tempering max temperature
                T_min 0.00001 #FLOAT32 parallel tempering min temperature
                time_limit {self.time_limit} #FLOAT64 time limit for search in seconds
                round_limit 100000000 #INT round/iteration limit for search. Search ends if no cost improvement found within a 10000 round window
                num_replicas_per_controller 32 #INT (POW2 only) number of replicas per parallel tempering controller
                num_controllers 1 #INT (POW2 only) number of parallel tempering controllers
                num_cores_per_controller {self.replicator_cores} #INT (POW2 only) number of cores/threads to dedicate to each controller
                ladder_init_mode 2 #INT (0,1,2) parallel tempering ladder init mode. 0->linear spacing b/w t_min & t_max. 1->linear spacing between beta_max and beta_min, then translated to T. 2->exponential spacing between T_min and T_max
                """

        params_path = self._dist_dir / "dist_matrix.params"
        params_path.write_text(param)

        replicator_path = DEFAULT_REPLICATOR_PATH
        if replicator_path.is_dir():
            candidate_paths = [
                replicator_path / "ReplicaTOR",
                replicator_path / "cmake-build" / "ReplicaTOR",
            ]
            replicator_path = next((p for p in candidate_paths if p.exists()), replicator_path)
        if not replicator_path.exists():
            raise FileNotFoundError(
                f"ReplicaTOR binary not found at {replicator_path}. Set REPLICATOR_PATH to the compiled binary."
            )
        if not os.access(replicator_path, os.X_OK):
            raise PermissionError(
                f"ReplicaTOR at {replicator_path} is not executable. Run 'chmod +x {replicator_path}' or update REPLICATOR_PATH."
            )

        soln_path = self._dist_dir / "dist_matrix.soln.txt"

        result = subprocess.run(
            [str(replicator_path), str(params_path)],
            check=True,
            capture_output=True,
            text=True,
            cwd=self._dist_dir,
        )

        full_output = result.stdout + "\n" + result.stderr

        #lire le résultat et le mettre en liste
        if not soln_path.exists():
            medoids = self._parse_medoids_from_output(full_output)
            if medoids:
                soln_path.write_text(" ".join(str(x) for x in medoids))
            else:
                raise FileNotFoundError(
                    "ReplicaTOR finished without writing dist_matrix.soln.txt; "
                    "ensuring the distance matrix contains no NaN values."
                )
        with open(soln_path, "r") as f:
            ligne = f.read()

        medoids = [int(x) for x in ligne.strip().split()]
        cluster_assignments = self._parse_cluster_assignments_from_output(full_output, n)
        return medoids, cluster_assignments

    @staticmethod
    def _parse_medoids_from_output(output: str):
        """Fallback parser for medoid indices if ReplicaTOR fails to emit soln file."""
        match = re.search(r"K Medoid Indices:\s*(.+?)\n\s*Cluster Assignments", output, re.S)
        if not match:
            return []
        numbers = re.findall(r"\d+", match.group(1))
        return [int(n) for n in numbers]

    @staticmethod
    def _parse_cluster_assignments_from_output(output: str, n: int):
        """Parse cluster assignments (0..K-1) for each of the n stocks from ReplicaTOR stdout.

        Expected format in ReplicaTOR output:
            Cluster Assignments (from 0 to K-1):
            0 2 1 0 3 ...  (N space-separated integers, one per stock)
        """
        match = re.search(r"Cluster Assignments[^\n]*:\s*\n([\d\s]+)", output)
        if not match:
            return None
        numbers = re.findall(r"\d+", match.group(1))
        if len(numbers) < n:
            print(f"Warning: expected {n} cluster assignments, got {len(numbers)}. Falling back to QP.")
            return None
        return [int(x) for x in numbers[:n]]

    def _qp_weights(self):
        """QP minimizing in-sample tracking error with sum=1 and mean-return constraints.

        The mean-return constraint forces E[r_portfolio] = E[r_index], eliminating any
        systematic bias (over/under-performance). The variance of the tracking error is
        then the sole minimization target, which is the correct formulation for index tracking.

        If SLSQP fails to converge with both constraints, falls back to sum=1 only (with warning).
        """
        subset_returns = self.stocks_returns[:, self.idx]
        index_returns = np.asarray(self.index_returns)

        valid_rows = np.isfinite(index_returns) & np.isfinite(subset_returns).all(axis=1)
        subset_returns = subset_returns[valid_rows]
        index_returns = index_returns[valid_rows]
        if subset_returns.size == 0:
            raise ValueError("No overlapping non-missing returns available for QUOB optimization.")

        initial_weight = np.ones(len(self.idx))
        initial_weight /= initial_weight.sum()
        bounds = [(0, 1) for _ in range(len(self.idx))]

        objective_function = lambda weight: np.sum((subset_returns @ weight - index_returns)**2)
        mu_stocks = subset_returns.mean(axis=0)
        mu_index = index_returns.mean()

        constraints_with_mean = [
            {'type': 'eq', 'fun': lambda weight: np.sum(weight) - 1},
            {'type': 'eq', 'fun': lambda weight: mu_stocks @ weight - mu_index},
        ]
        result = minimize(objective_function, initial_weight, method='SLSQP',
                          constraints=constraints_with_mean, bounds=bounds)

        if not result.success:
            print(f"Warning: SLSQP did not converge with mean constraint ({result.message}); "
                  "retrying with sum=1 only.")
            constraints_fallback = [{'type': 'eq', 'fun': lambda weight: np.sum(weight) - 1}]
            result = minimize(objective_function, initial_weight, method='SLSQP',
                              constraints=constraints_fallback, bounds=bounds)
            if not result.success:
                print(f"Warning: SLSQP fallback also did not converge: {result.message}")

        # Renormalize defensively in case the optimizer drifts from the sum=1 constraint.
        weights = result.x
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights / weight_sum
        return weights

    def calc_weights(self):
        self.idx, self.cluster_assignments = self.stock_picking()
        return self._qp_weights()

    def get_weights(self):
        #retourne numpy array sparse des poids
        weight_global = np.zeros(self.stocks_returns.shape[1])
        try:
            micro_weight = self.calc_weights()
        finally:
            # Clean up the per-instance temp directory now that the solve is complete.
            self._cleanup()
        for i in range(len(micro_weight)):
            weight_global[self.idx[i]] = micro_weight[i]

        return weight_global
