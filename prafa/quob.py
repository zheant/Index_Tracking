from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Final

import dcor
import numpy as np
from scipy.optimize import minimize


def _resolve_replicator_bin(explicit: os.PathLike[str] | str | None = None) -> Path:
    """Return a usable ReplicaTOR binary path.

    The lookup order favours an explicit CLI argument, then the ``REPLICATOR_BIN``
    environment variable, followed by a couple of common build locations that are
    used throughout the project documentation.  The first existing, executable file
    is returned; otherwise a :class:`FileNotFoundError` is raised.
    """

    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit))

    env_value = os.environ.get("REPLICATOR_BIN")
    if env_value:
        candidates.append(Path(env_value))

    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            Path.home() / "or_tool/cmake-build/ReplicaTOR",
            Path.home() / "or_tool/ReplicaTOR/cmake-build/ReplicaTOR",
            repo_root / "or_tool/cmake-build/ReplicaTOR",
            repo_root / "or_tool/ReplicaTOR/cmake-build/ReplicaTOR",
        ]
    )

    checked = []
    for candidate in candidates:
        path = candidate.expanduser()
        checked.append(path)
        if path.is_file() and os.access(path, os.X_OK):
            return path
        # Some users forget to add the execute bit after compiling; detect this to
        # provide a clearer error message.
        if path.is_file():
            current_mode = stat.S_IMODE(path.stat().st_mode)
            if not (current_mode & stat.S_IXUSR):
                raise PermissionError(
                    "ReplicaTOR binary trouvé mais non exécutable à l'emplacement "
                    f"{path}. Ajoutez les permissions d'exécution (chmod +x)."
                )

    searched = "\n - ".join(str(p) for p in checked)
    raise FileNotFoundError(
        "Impossible de localiser le binaire ReplicaTOR. Chemins vérifiés :\n"
        f" - {searched}\n"
        "Fournissez un chemin valide via --replicator-bin ou la variable d'environnement "
        "REPLICATOR_BIN."
    )


def _validate_replicator_cores(value: int) -> int:
    """Ensure the requested ReplicaTOR core count is a positive power of two."""

    if value <= 0:
        raise ValueError("replicator_cores doit être un entier strictement positif")

    if value & (value - 1) != 0:
        raise ValueError(
            "replicator_cores doit être une puissance de deux (1, 2, 4, ...)."
        )

    return value


class QUOB:
    def __init__(
        self,
        stocks_returns,
        index_returns,
        K,
        simple_corr=False,
        replicator_bin: os.PathLike[str] | str | None = None,
        replicator_time_limit: float = 300.0,
        replicator_cores: int = 1,
    ):
        #matrice et vecteur numpy
        self.stocks_returns = stocks_returns
        self.index_returns = index_returns
        self.K = K #cardinalité!!
        self.idx = None #liste d'indice des stonks choisit
        self.distance_matrix: np.ndarray | None = None
        self.dist_dir = Path(__file__).resolve().parent / "dist_matrix"
        self.dist_dir.mkdir(parents=True, exist_ok=True)
        self.replicator_bin = _resolve_replicator_bin(replicator_bin)
        self.replicator_time_limit = float(replicator_time_limit)
        self.problem_name = "dist_matrix"
        self.replicator_cores: Final[int] = _validate_replicator_cores(int(replicator_cores))

        #construire ma matrice de distance
        if simple_corr:
            self.matrix_simplecor()
        else:
            self.matrix_dcor()

        self._write_adjacency_matrix()
        


    def matrix_dcor(self):
        
        Welsch_function = lambda x : 1 - np.exp(-0.5 * x)

        n = self.stocks_returns.shape[1]
        dcor_mat = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                dcor_val = dcor.distance_correlation(self.stocks_returns[:, i], self.stocks_returns[:, j])
                dist = 1 - dcor_val
                dcor_mat[i, j] = dcor_mat[j, i] = Welsch_function(dist) #Welsch_function(dist)
    
        self.distance_matrix = dcor_mat
        matrix_path = self.dist_dir / f"{self.problem_name}.d"
        np.savetxt(matrix_path, dcor_mat)



    def matrix_simplecor(self):
        distance_func = lambda di : np.sqrt(0.5*(1 - di))
        Welsch_function = lambda x : 1 - np.exp(-0.5 * x)

        n = self.stocks_returns.shape[1]
        corr_matrix = np.corrcoef(self.stocks_returns, rowvar=False)

        distance_matrix = distance_func(corr_matrix)
        self.distance_matrix = Welsch_function(distance_matrix)
        matrix_path = self.dist_dir / f"{self.problem_name}.d"
        np.savetxt(matrix_path, self.distance_matrix)

    def _write_adjacency_matrix(self) -> None:
        """Persist a fully connected adjacency matrix for ReplicaTOR.

        The k-medoids formulation solved by ReplicaTOR expects an ``.adj`` file
        that describes the graph structure associated with the distance matrix.
        The optimisation we run is over a complete graph, therefore the
        adjacency matrix simply contains ones everywhere except on the diagonal
        (no self-loops).  Writing the file explicitly avoids runtime failures
        when ReplicaTOR loads the problem definition.
        """

        n = self.stocks_returns.shape[1]
        adjacency = np.ones((n, n), dtype=int)
        np.fill_diagonal(adjacency, 0)

        adjacency_path = self.dist_dir / f"{self.problem_name}.adj"
        np.savetxt(adjacency_path, adjacency, fmt="%d")


    def stock_picking(self, n):
        #résolution du probleme d'optimisation
        #retourne une liste d'indice des stonks sélectionné
        matrix_stem = self.dist_dir / self.problem_name
        param = f"""num_vars {n} #INT number of variables/nodes
                num_k {self.K} #INT number of medoids/exemplars
                B_scale_factor {0.0333} 0.5*(self.K+1)/n#FLOAT32 scaling factor for model bias, set to 0.5*(num_k +1)/num_vars
                D_scale_factor 1.0 #FLOAT32 scaling factor for model distances, leave at 1
                problem_path {self.dist_dir.as_posix()}/
                problem_name {self.problem_name}
                cost_answer -1000000 #FLOAT32 target cost to allow program to exit early if found, set to large neg value if you don't want an early exit
                T_max 0.01 #FLOAT32 parallel tempering max temperature
                T_min 0.00001 #FLOAT32 parallel tempering min temperature
                time_limit {self.replicator_time_limit} #FLOAT64 time limit for search in seconds
                round_limit 100000000 #INT round/iteration limit for search. Search ends if no cost improvement found within a 10000 round window
                num_replicas_per_controller 32 #INT (POW2 only) number of replicas per parallel tempering controller
                num_controllers 1 #INT (POW2 only) number of parallel tempering controllers
                num_cores_per_controller {self.replicator_cores} #INT (POW2 only) number of cores/threads to dedicate to each controller
                ladder_init_mode 2 #INT (0,1,2) parallel tempering ladder init mode. 0->linear spacing b/w t_min & t_max. 1->linear spacing between beta_max and beta_min, then translated to T. 2->exponential spacing between T_min and T_max
                """

        params_path = matrix_stem.with_suffix(".params")
        with open(params_path, "w", encoding="utf-8") as f:
            f.write(param)

        env = os.environ.copy()
        # Several ReplicaTOR builds rely on OpenMP for intra-controller parallelism.
        # Enforcing ``OMP_NUM_THREADS`` ensures the solver honours the requested
        # ``num_cores_per_controller`` value instead of silently defaulting to a
        # single CPU even when the parameter file requests more threads.
        env["OMP_NUM_THREADS"] = str(self.replicator_cores)
        # Some builds also look at this environment variable; setting it is
        # harmless when unsupported and keeps behaviour consistent.
        env["REPLICATOR_NUM_CORES"] = str(self.replicator_cores)

        completed = subprocess.run(
            [self.replicator_bin.as_posix(), params_path.as_posix()],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        if completed.stdout:
            print(completed.stdout, end="")
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)

        #lire le résultat et le mettre en liste
        solution_path = matrix_stem.with_suffix(".soln.txt")
        legacy_solution_path = matrix_stem.with_suffix(".soln")

        medoids: list[int] | None = None

        if solution_path.exists():
            with open(solution_path, "r", encoding="utf-8") as f:
                ligne = f.read()
            medoids = [int(x) for x in ligne.strip().split() if x]
        elif legacy_solution_path.exists():
            with open(legacy_solution_path, "r", encoding="utf-8") as f:
                ligne = f.read()
            medoids = [int(x) for x in ligne.strip().split() if x]
        else:
            medoids = self._parse_medoids_from_stdout(completed.stdout)

        if medoids:
            if completed.returncode != 0:
                message = (
                    "ReplicaTOR s'est terminé de manière inattendue (code de sortie "
                    f"{completed.returncode}), mais les indices de médianoïdes ont pu "
                    "être récupérés depuis la sortie standard."
                )
                print(message, file=sys.stderr)
        else:
            available = ", ".join(
                sorted(str(path.name) for path in self.dist_dir.glob("dist_matrix.soln*"))
            )
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(
                    completed.returncode,
                    completed.args,
                    output=completed.stdout,
                    stderr=completed.stderr,
                )

            raise FileNotFoundError(
                "ReplicaTOR n'a produit aucun fichier de solution attendu et la sortie ne "
                "contient pas d'indices de médianoïdes. Fichiers disponibles : "
                f"{available or 'aucun'}"
            )

        if self.distance_matrix is None:
            self.distance_matrix = np.loadtxt(matrix_stem.with_suffix(".d"))

        cluster_assignments = np.argmin(self.distance_matrix[:, medoids], axis=1)
        cluster_path = matrix_stem.with_suffix(".clusters.txt")
        np.savetxt(cluster_path, cluster_assignments, fmt="%d")

        return medoids

    @staticmethod
    def _parse_medoids_from_stdout(stdout: str | None) -> list[int] | None:
        if not stdout:
            return None

        lines = stdout.splitlines()
        collecting = False
        tokens: list[str] = []

        for line in lines:
            stripped = line.strip()

            if not collecting:
                if stripped.startswith("K Medoid Indices"):
                    collecting = True
                continue

            if not stripped:
                continue

            if stripped.startswith("Cluster Assignments"):
                break

            tokens.extend(stripped.split())

        if not tokens:
            return None

        try:
            return [int(token) for token in tokens]
        except ValueError:
            return None
        return None


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
