import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import subprocess
import re

from prafa.matrix_utils import compute_dcor_matrix, compute_simplecor_matrix


DEFAULT_REPLICATOR_PATH = Path(
    os.environ.get("REPLICATOR_PATH", Path.home() / "or_tool" / "ReplicaTOR" / "cmake-build")
)
DEFAULT_REPLICATOR_CORES = int(os.environ.get("REPLICATOR_CORES", 8))
DEFAULT_TIME_LIMIT = 300.0


class QUOB:
    """Wrapper around the ReplicaTOR binary for k-medoid selection.

    Selects K representative stocks (medoids) from the universe using the
    ReplicaTOR parallel-tempering solver, then fits portfolio weights via
    a constrained quadratic program that minimises in-sample tracking error.

    Parameters
    ----------
    stocks_returns:
        Array of shape (T, n) — training-period returns for n stocks.
    index_returns:
        Array of shape (T,) — training-period index returns.
    K:
        Target number of medoids (cardinality).
    simple_corr:
        If True, use Pearson-based distance; otherwise use distance correlation.
    replicator_cores:
        Number of OpenMP threads for ReplicaTOR.
    time_limit:
        Wall-clock time limit for the ReplicaTOR search (seconds).
    d_scale:
        D_scale_factor for ReplicaTOR.  Controls the weight of the dispersion
        term relative to the centrality term in the objective.
    precomputed_dist:
        Pre-computed distance matrix of shape (n, n).  When provided, skips
        distance computation (useful when the matrix was already built for
        Neyman allocation or similar purposes).
    """

    def __init__(
        self,
        stocks_returns: np.ndarray,
        index_returns: np.ndarray,
        K: int,
        simple_corr: bool = False,
        replicator_cores: int | None = None,
        time_limit: float | None = None,
        d_scale: float = 1.0,
        precomputed_dist: np.ndarray | None = None,
    ):
        self.stocks_returns = stocks_returns
        self.index_returns = index_returns
        self.K = K
        self.idx: list[int] | None = None
        self.cluster_assignments: list[int] | None = None
        self.d_scale = d_scale
        self.replicator_cores = replicator_cores or DEFAULT_REPLICATOR_CORES
        self.time_limit = time_limit if time_limit is not None else DEFAULT_TIME_LIMIT

        self._dist_dir = Path(tempfile.mkdtemp(prefix="quob_"))

        if precomputed_dist is not None:
            n = precomputed_dist.shape[0]
            np.savetxt(self._dist_dir / "dist_matrix.d", precomputed_dist)
            adj = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
            np.savetxt(self._dist_dir / "dist_matrix.adj", adj, fmt="%d")
        elif simple_corr:
            self._write_distance_matrix(compute_simplecor_matrix(stocks_returns))
        else:
            self._write_distance_matrix(compute_dcor_matrix(stocks_returns))

    def _write_distance_matrix(self, mat: np.ndarray) -> None:
        n = mat.shape[0]
        np.savetxt(self._dist_dir / "dist_matrix.d", mat)
        adj = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
        np.savetxt(self._dist_dir / "dist_matrix.adj", adj, fmt="%d")

    def _cleanup(self) -> None:
        if self._dist_dir.exists():
            shutil.rmtree(self._dist_dir, ignore_errors=True)

    def stock_picking(self) -> tuple[list[int], list[int] | None]:
        """Run ReplicaTOR and return (medoid_indices, cluster_assignments)."""
        n = self.stocks_returns.shape[1]
        b_scale = 0.5 * (self.K + 1) / n
        param = f"""num_vars {n} #INT number of variables/nodes
                num_k {self.K} #INT number of medoids/exemplars
                B_scale_factor {b_scale} #FLOAT32 scaling factor for model bias, set to 0.5*(num_k+1)/num_vars
                D_scale_factor {self.d_scale} #FLOAT32 scaling factor for model distances
                problem_path {self._dist_dir}/
                problem_name dist_matrix
                cost_answer -1000000 #FLOAT32 target cost for early exit
                T_max 0.01 #FLOAT32 parallel tempering max temperature
                T_min 0.00001 #FLOAT32 parallel tempering min temperature
                time_limit {self.time_limit} #FLOAT64 time limit in seconds
                round_limit 100000000 #INT iteration limit
                num_replicas_per_controller 32 #INT (POW2 only)
                num_controllers 1 #INT (POW2 only)
                num_cores_per_controller {self.replicator_cores} #INT (POW2 only)
                ladder_init_mode 2 #INT exponential spacing between T_min and T_max
                """

        params_path = self._dist_dir / "dist_matrix.params"
        params_path.write_text(param)

        replicator_path = DEFAULT_REPLICATOR_PATH
        if replicator_path.is_dir():
            candidates = [
                replicator_path / "ReplicaTOR",
                replicator_path / "cmake-build" / "ReplicaTOR",
            ]
            replicator_path = next((p for p in candidates if p.exists()), replicator_path)
        if not replicator_path.exists():
            raise FileNotFoundError(
                f"ReplicaTOR binary not found at {replicator_path}. "
                "Set REPLICATOR_PATH to the compiled binary."
            )
        if not os.access(replicator_path, os.X_OK):
            raise PermissionError(
                f"ReplicaTOR at {replicator_path} is not executable. "
                f"Run 'chmod +x {replicator_path}' or update REPLICATOR_PATH."
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

        if not soln_path.exists():
            medoids = self._parse_medoids_from_output(full_output)
            if medoids:
                soln_path.write_text(" ".join(str(x) for x in medoids))
            else:
                raise FileNotFoundError(
                    "ReplicaTOR finished without writing dist_matrix.soln.txt. "
                    "Ensure the distance matrix contains no NaN values."
                )

        with open(soln_path) as f:
            medoids = [int(x) for x in f.read().strip().split()]

        cluster_assignments = self._parse_cluster_assignments_from_output(full_output, n)
        return medoids, cluster_assignments

    @staticmethod
    def _parse_medoids_from_output(output: str) -> list[int]:
        """Fallback parser for medoid indices when ReplicaTOR omits the soln file."""
        match = re.search(r"K Medoid Indices:\s*(.+?)\n\s*Cluster Assignments", output, re.S)
        if not match:
            return []
        return [int(x) for x in re.findall(r"\d+", match.group(1))]

    @staticmethod
    def _parse_cluster_assignments_from_output(output: str, n: int) -> list[int] | None:
        """Parse per-stock cluster assignments (0..K-1) from ReplicaTOR stdout."""
        match = re.search(r"Cluster Assignments[^\n]*:\s*\n([\d\s]+)", output)
        if not match:
            return None
        numbers = re.findall(r"\d+", match.group(1))
        if len(numbers) < n:
            print(f"Warning: expected {n} cluster assignments, got {len(numbers)}. Falling back to nearest-medoid assignment.")
            return None
        return [int(x) for x in numbers[:n]]

    def _qp_weights(self) -> np.ndarray:
        """Fit portfolio weights by minimising in-sample tracking error variance.

        Solves:
            min  Var(R_selected @ w - r_index)
            s.t. sum(w) = 1,  E[R_selected @ w] = E[r_index],  w >= 0

        The mean-return constraint eliminates systematic bias.  If SLSQP fails
        with both constraints, retries with sum=1 only.
        """
        subset_returns = self.stocks_returns[:, self.idx]
        index_returns = np.asarray(self.index_returns)

        valid = np.isfinite(index_returns) & np.isfinite(subset_returns).all(axis=1)
        subset_returns = subset_returns[valid]
        index_returns = index_returns[valid]
        if subset_returns.size == 0:
            raise ValueError("No overlapping non-missing returns available for QP optimisation.")

        K = len(self.idx)
        w0 = np.ones(K) / K
        bounds = [(0, 1)] * K
        objective = lambda w: np.sum((subset_returns @ w - index_returns) ** 2)

        mu_s = subset_returns.mean(axis=0)
        mu_i = index_returns.mean()
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1},
            {"type": "eq", "fun": lambda w: mu_s @ w - mu_i},
        ]
        result = minimize(objective, w0, method="SLSQP", constraints=constraints, bounds=bounds)

        if not result.success:
            print(f"Warning: SLSQP did not converge with mean constraint ({result.message}). "
                  "Retrying with sum=1 only.")
            result = minimize(objective, w0, method="SLSQP",
                              constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                              bounds=bounds)
            if not result.success:
                print(f"Warning: SLSQP fallback also did not converge: {result.message}")

        weights = result.x
        total = weights.sum()
        return weights / total if total > 0 else weights

    def get_weights(self) -> np.ndarray:
        """Run the full pipeline: medoid selection → QP weighting → sparse global weights."""
        weight_global = np.zeros(self.stocks_returns.shape[1])
        try:
            self.idx, self.cluster_assignments = self.stock_picking()
            micro_weight = self._qp_weights()
        finally:
            self._cleanup()
        for i, idx in enumerate(self.idx):
            weight_global[idx] = micro_weight[i]
        return weight_global
