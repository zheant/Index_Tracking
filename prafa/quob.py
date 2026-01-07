import os
from pathlib import Path
import dcor
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import subprocess
import re



DIST_DIR = Path(__file__).resolve().parent / "dist_matrix"
# Default to the compiled ReplicaTOR binary under ~/or_tool/ReplicaTOR/cmake-build unless overridden.
DEFAULT_REPLICATOR_PATH = Path(
    os.environ.get("REPLICATOR_PATH", Path.home() / "or_tool" / "ReplicaTOR" / "cmake-build")
)
DEFAULT_REPLICATOR_CORES = int(os.environ.get("REPLICATOR_CORES", 8))
DEFAULT_TIME_LIMIT = 300.0


class QUOB:
    def __init__(self, stocks_returns, index_returns, K, simple_corr=False, replicator_cores=None, time_limit=None):
        #matrice et vecteur numpy
        self.stocks_returns = stocks_returns
        self.index_returns = index_returns
        self.K = K #cardinalité!!
        self.idx = None #liste d'indice des stonks choisit
        self.replicator_cores = replicator_cores or DEFAULT_REPLICATOR_CORES
        self.time_limit = DEFAULT_TIME_LIMIT if time_limit is None else time_limit

        # Always start from a clean distance directory to avoid stale params/solutions.
        self._prepare_dist_dir()
        
        #construire ma matrice de distance
        if simple_corr:
            self.matrix_simplecor()
        else:
            self.matrix_dcor()
        


    def _prepare_dist_dir(self):
        DIST_DIR.mkdir(parents=True, exist_ok=True)
        for pattern in ("dist_matrix.d", "dist_matrix.adj", "dist_matrix.soln.txt", "dist_matrix.params"):
            path = DIST_DIR / pattern
            if path.exists():
                path.unlink()

    def matrix_dcor(self):

        Welsch_function = lambda x : 1 - np.exp(-0.5 * x)

        n = self.stocks_returns.shape[1]
        dcor_mat = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                dcor_val = dcor.distance_correlation(self.stocks_returns[:, i], self.stocks_returns[:, j])
                dist = 1 - dcor_val
                dcor_mat[i, j] = dcor_mat[j, i] = Welsch_function(dist) #Welsch_function(dist)

        dcor_mat = np.nan_to_num(dcor_mat, nan=1.0, posinf=1.0, neginf=1.0)
        np.fill_diagonal(dcor_mat, 0.0)
        dcor_mat = np.clip(dcor_mat, 0.0, 1.0)
        np.savetxt(DIST_DIR / "dist_matrix.d", dcor_mat)
        adj = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
        np.savetxt(DIST_DIR / "dist_matrix.adj", adj, fmt="%d")
        


    def matrix_simplecor(self):
        distance_func = lambda di : np.sqrt(0.5*(1 - di))
        Welsch_function = lambda x : 1 - np.exp(-0.5 * x)

        n = self.stocks_returns.shape[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_matrix = np.corrcoef(self.stocks_returns, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

        distance_matrix = distance_func(corr_matrix)
        distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=1.0, neginf=1.0)
        distance_matrix = np.nan_to_num(Welsch_function(distance_matrix), nan=1.0, posinf=1.0, neginf=1.0)
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = np.clip(distance_matrix, 0.0, 1.0)

        np.savetxt(DIST_DIR / "dist_matrix.d", distance_matrix)
        adj = np.ones((n, n), dtype=int) - np.eye(n, dtype=int)
        np.savetxt(DIST_DIR / "dist_matrix.adj", adj, fmt="%d")


    def stock_picking(self, n):
        #résolution du probleme d'optimisation 
        #retourne une liste d'indice des stonks sélectionné
        param = f"""num_vars {n} #INT number of variables/nodes
                num_k {self.K} #INT number of medoids/exemplars
                B_scale_factor {0.0333} 0.5*(self.K+1)/n#FLOAT32 scaling factor for model bias, set to 0.5*(num_k +1)/num_vars
                D_scale_factor 1.0 #FLOAT32 scaling factor for model distances, leave at 1
                problem_path {DIST_DIR}/
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

        params_path = DIST_DIR / "dist_matrix.params"
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

        soln_path = DIST_DIR / "dist_matrix.soln.txt"
        if soln_path.exists():
            soln_path.unlink()

        result = subprocess.run(
            [str(replicator_path), str(params_path)],
            check=True,
            capture_output=True,
            text=True,
            cwd=DIST_DIR,
        )

        #lire le résultat et le mettre en liste
        if not soln_path.exists():
            medoids = self._parse_medoids_from_output(result.stdout + "\n" + result.stderr)
            if medoids:
                soln_path.write_text(" ".join(str(x) for x in medoids))
            else:
                raise FileNotFoundError(
                    "ReplicaTOR finished without writing dist_matrix.soln.txt; remove stale dist_matrix files and rerun, "
                    "ensuring the distance matrix contains no NaN values."
                )
        with open(soln_path, "r") as f:
            ligne = f.read()

        return [int(x) for x in ligne.strip().split()]

    @staticmethod
    def _parse_medoids_from_output(output: str):
        """Fallback parser for medoid indices if ReplicaTOR fails to emit soln file."""
        match = re.search(r"K Medoid Indices:\s*(.+?)\n\s*Cluster Assignments", output, re.S)
        if not match:
            return []
        numbers = re.findall(r"\d+", match.group(1))
        return [int(n) for n in numbers]


    def calc_weights(self):
        self.idx = self.stock_picking(self.stocks_returns.shape[1])
        subset_returns = self.stocks_returns[:, self.idx]
        
        initial_weight = np.ones(len(self.idx))
        initial_weight /= initial_weight.sum()  
        bounds = [(0, 1) for _ in range(len(self.idx))]

        # Define Constraints    
        constraint = {'type': 'eq', 'fun':lambda weight : np.sum(weight) - 1}
        objective_function = lambda weight : np.sum((subset_returns @ weight - self.index_returns)**2)
        
        # Optimization
        result = minimize(objective_function, initial_weight, method = 'SLSQP', constraints=constraint, bounds=bounds)
        return result.x

    
    def get_weights(self):
        #retourne numpy array sparse des poids
        
        weight_global = np.zeros(self.stocks_returns.shape[1])

        micro_weight = self.calc_weights()
        for i in range(len(micro_weight)):
            weight_global[self.idx[i]] = micro_weight[i]

        return weight_global
