from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Final

import dcor
import gurobipy as gp
import numpy as np
from gurobipy import GRB
from numpy import typing as npt
from scipy.optimize import minimize

from prafa.quob import MAX_DCOR_ASSETS


def _status_name(status: int) -> str:
    """Retourne le libellé lisible associé au code de statut Gurobi."""

    for name, value in GRB.Status.__dict__.items():
        if not name.startswith("_") and isinstance(value, int) and value == status:
            return name
    return str(status)


def _welsch(x: npt.ArrayLike) -> np.ndarray:
    """Applique la transformation de Welsch pour atténuer les valeurs extrêmes."""

    arr = np.asarray(x, dtype=float)
    return 1.0 - np.exp(-0.5 * arr)


class Gurobi:
    """Résout le problème d'index-tracking via la formulation binaire quadratique."""

    def __init__(
        self,
        stocks_returns: npt.NDArray[np.float64],
        index_returns: npt.NDArray[np.float64],
        K: int,
        simple_corr: bool = False,
        time_limit: float = 10800.0,
        log_dir: str | None = None,
        log_label: str | None = None,
    ) -> None:
        if K <= 0:
            raise ValueError("La cardinalité doit être strictement positive.")

        self.stocks_returns = np.asarray(stocks_returns, dtype=float)
        self.index_returns = np.asarray(index_returns, dtype=float)
        self.K = int(K)
        self.simple_corr = bool(simple_corr)
        self.time_limit: Final[float] = float(time_limit)

        self.idx: list[int] | None = None
        self.distance_matrix: np.ndarray | None = None
        self._env = self._create_env()
        self.last_runtime: float | None = None
        self.last_status: str | None = None
        self.log_path = self._prepare_log_path(log_dir, log_label)

    # ------------------------------------------------------------------
    # Matrices de distance
    # ------------------------------------------------------------------
    def matrix_dcor(self) -> np.ndarray:
        """Construit la matrice des distances via la distance de corrélation."""

        n_samples, n_assets = self.stocks_returns.shape

        if n_assets == 0:
            raise ValueError(
                "Impossible de calculer la matrice de distance : aucun titre disponible dans la fenêtre."
            )
        if n_samples < 2:
            raise ValueError(
                "La fenêtre d'entraînement ne contient pas suffisamment d'observations."
            )

        if n_assets > MAX_DCOR_ASSETS:
            print(
                "⚠️ Taille de l'univers trop importante pour dcor "
                f"({n_assets} titres). Passage à la corrélation de Pearson."
            )
            distance_matrix = self._compute_corr_distance()
        else:
            try:
                distance_matrix = self._compute_dcor_matrix(n_assets)
            except Exception as exc:  # pragma: no cover - dépend du backend dcor
                print(
                    "⚠️ Échec du calcul de la distance de corrélation (dcor). Passage à la corrélation de Pearson.",
                    f"Raison : {exc}",
                )
                distance_matrix = self._compute_corr_distance()
            else:
                distance_matrix = _welsch(distance_matrix)

        self.distance_matrix = distance_matrix
        return distance_matrix

    def _compute_dcor_matrix(self, n_assets: int) -> np.ndarray:
        output = np.zeros((n_assets, n_assets), dtype=float)
        progress_step = max(1, n_assets // 10)

        for i in range(n_assets):
            if n_assets >= 100 and i % progress_step == 0:
                print(f"  • Calcul des corrélations de distance : ligne {i + 1}/{n_assets}")
            for j in range(i, n_assets):
                dcor_val = dcor.distance_correlation(
                    self.stocks_returns[:, i],
                    self.stocks_returns[:, j],
                )
                dist = 1.0 - dcor_val
                output[i, j] = output[j, i] = dist

        return output

    def _compute_corr_distance(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_matrix = np.corrcoef(self.stocks_returns, rowvar=False)

        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        distance = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr_matrix)))
        return _welsch(distance)

    def matrix_simplecor(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            corr_matrix = np.corrcoef(self.stocks_returns, rowvar=False)

        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        distance = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr_matrix)))
        self.distance_matrix = _welsch(distance)
        return self.distance_matrix

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------
    def stock_picking(self, n_assets: int) -> np.ndarray:
        if self.simple_corr:
            distance_matrix = self.matrix_simplecor()
        else:
            distance_matrix = self.matrix_dcor()

        n = distance_matrix.shape[0]
        if n == 0:
            raise ValueError("Aucun titre disponible pour l'optimisation.")
        if self.K > n:
            raise ValueError(
                f"La cardinalité demandée ({self.K}) dépasse le nombre de titres disponibles ({n})."
            )

        alpha = 1.0 / self.K
        beta = 1.0 / n
        ones = np.ones(n)
        c = beta * (distance_matrix @ ones)

        try:
            model = gp.Model("BQO_compact", env=self._env)
        except gp.GurobiError as exc:  # pragma: no cover - dépend de l'installation Gurobi
            raise RuntimeError(
                "Impossible de démarrer Gurobi. Vérifiez votre licence et l'installation de gurobipy."
            ) from exc

        model.Params.OutputFlag = 1
        model.Params.TimeLimit = self.time_limit
        model.Params.NonConvex = 2
        if self.log_path is not None:
            model.Params.LogFile = self.log_path.as_posix()

        z = model.addMVar(n, vtype=GRB.BINARY, name="z")

        objective = c @ z - 0.5 * alpha * (z @ distance_matrix @ z)
        model.setObjective(objective, GRB.MINIMIZE)
        model.addConstr(ones @ z == self.K, name="card")

        print(
            "▶️ Lancement de Gurobi :",
            f"limite {self.time_limit / 3600:.2f} h, {n} titres, cardinalité {self.K}",
        )

        try:
            try:
                model.optimize()
            except gp.GurobiError as exc:  # pragma: no cover - dépend de Gurobi
                raise RuntimeError(f"Gurobi a échoué durant l'optimisation : {exc}") from exc

            status = model.Status
            status_name = _status_name(status)
            self.last_status = status_name
            self.last_runtime = getattr(model, "Runtime", None)

            if status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
                if model.SolCount == 0:
                    raise RuntimeError(
                        "Gurobi n'a retourné aucune solution exploitable malgré le statut de fin obtenu."
                    )
                if status == GRB.TIME_LIMIT:
                    print(
                        "⚠️ Gurobi a atteint la limite de temps mais une solution réalisable a été conservée."
                    )
                elif status == GRB.SUBOPTIMAL:
                    print(
                        "⚠️ Gurobi a terminé avec un statut suboptimal. Utilisation de la meilleure solution disponible."
                    )
            else:
                raise RuntimeError(
                    "Gurobi s'est arrêté avec le statut "
                    f"{status_name}. Consultez les logs ci-dessus pour plus de détails."
                )

            runtime_msg = (
                f"{self.last_runtime / 3600:.2f} h"
                if self.last_runtime is not None
                else "durée inconnue"
            )
            best_obj = model.ObjVal if model.SolCount > 0 else float("nan")
            print(
                "⏱️ Gurobi terminé :",
                status_name,
                f"en {runtime_msg}",
                f"(objectif {best_obj:.6f})",
            )
            if self.log_path is not None:
                print(f"📄 Journal complet : {self.log_path}")

            solution = z.X.copy()
            return solution
        finally:
            model.dispose()

    def calc_weights(self) -> np.ndarray:
        stock_pick_binary = self.stock_picking(self.stocks_returns.shape[1])
        selected = np.where(stock_pick_binary > 0.5)[0]
        if selected.size == 0:
            raise RuntimeError(
                "Gurobi n'a sélectionné aucun titre. Vérifiez les journaux de résolution pour identifier la cause."
            )

        self.idx = selected.tolist()
        subset_returns = self.stocks_returns[:, self.idx]

        initial_weight = np.full(len(self.idx), 1.0 / len(self.idx))
        bounds = [(0.0, 1.0) for _ in self.idx]
        constraint = {"type": "eq", "fun": lambda weight: np.sum(weight) - 1.0}

        objective_function = lambda weight: np.sum((subset_returns @ weight - self.index_returns) ** 2)

        result = minimize(
            objective_function,
            initial_weight,
            method="SLSQP",
            constraints=constraint,
            bounds=bounds,
        )

        if not result.success:
            raise RuntimeError(
                "L'optimisation continue des poids a échoué : "
                f"{result.message}."
            )

        return result.x

    def get_weights(self) -> np.ndarray:
        weight_global = np.zeros(self.stocks_returns.shape[1])
        micro_weight = self.calc_weights()
        for local_idx, global_idx in enumerate(self.idx or []):
            weight_global[global_idx] = micro_weight[local_idx]

        try:
            return weight_global
        finally:  # pragma: no cover - dispose peut échouer selon l'environnement
            try:
                self._env.dispose()
            except gp.GurobiError:
                pass

    # ------------------------------------------------------------------
    # Gestion de l'environnement Gurobi
    # ------------------------------------------------------------------
    def _create_env(self) -> gp.Env:
        try:
            env = gp.Env(empty=True)
        except gp.GurobiError as exc:  # pragma: no cover - dépend de l'installation Gurobi
            raise RuntimeError(
                "Impossible d'initialiser l'environnement Gurobi. Vérifiez votre installation."
            ) from exc

        access_id = os.environ.get("GRB_WLSACCESSID")
        secret_key = os.environ.get("GRB_WLSSECRET")
        license_id = os.environ.get("GRB_LICENSEID")

        if any((access_id, secret_key, license_id)):
            missing = [
                name
                for name, value in (
                    ("GRB_WLSACCESSID", access_id),
                    ("GRB_WLSSECRET", secret_key),
                    ("GRB_LICENSEID", license_id),
                )
                if not value
            ]
            if missing:
                raise RuntimeError(
                    "La configuration WLS est incomplète. Variables manquantes : "
                    + ", ".join(missing)
                )

            env.setParam("WLSACCESSID", access_id)
            env.setParam("WLSSECRET", secret_key)
            try:
                env.setParam("LICENSEID", int(license_id))
            except ValueError:
                env.setParam("LICENSEID", license_id)

        try:
            env.start()
        except gp.GurobiError as exc:  # pragma: no cover - dépend de l'installation Gurobi
            raise RuntimeError(
                "Gurobi n'a pas pu démarrer l'environnement. Vérifiez vos identifiants de licence."
            ) from exc

        return env

    def _prepare_log_path(self, log_dir: str | None, log_label: str | None) -> Path | None:
        if not log_dir:
            return None

        base = Path(log_dir).expanduser()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = (log_label or "window").replace("/", "-").replace(" ", "_")
        filename = f"gurobi_{label}_{timestamp}.log"

        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None

        return base / filename
