"""Shared distance-matrix computations used by both QUOB and Gurobi solvers."""

import dcor
import numpy as np
import pandas as pd


def compute_dcor_matrix(stocks_returns: np.ndarray, min_obs: int = 500) -> np.ndarray:
    """Compute a Welsch-transformed distance-correlation matrix.

    Parameters
    ----------
    stocks_returns:
        2-D array of shape (T, n) where T is the number of observations and
        n is the number of assets.
    min_obs:
        Minimum number of jointly finite observations required to compute the
        distance correlation.  Pairs with fewer observations are assigned the
        maximum distance (1.0).

    Returns
    -------
    np.ndarray of shape (n, n) with values in [0, 1].  Diagonal is 0.
    """
    welsch = lambda x: 1 - np.exp(-0.5 * x)
    n = stocks_returns.shape[1]
    dcor_mat = np.zeros((n, n))

    for i in range(n):
        series_i = stocks_returns[:, i]
        for j in range(i, n):
            series_j = stocks_returns[:, j]
            mask = np.isfinite(series_i) & np.isfinite(series_j)
            if mask.sum() < min_obs:
                dist = 1.0
            else:
                dcor_val = dcor.distance_correlation(series_i[mask], series_j[mask])
                dist = 1 - dcor_val
            dcor_mat[i, j] = dcor_mat[j, i] = welsch(dist)

    dcor_mat = np.nan_to_num(dcor_mat, nan=1.0, posinf=1.0, neginf=1.0)
    np.fill_diagonal(dcor_mat, 0.0)
    return np.clip(dcor_mat, 0.0, 1.0)


def compute_simplecor_matrix(stocks_returns: np.ndarray, min_obs: int = 500) -> np.ndarray:
    """Compute a Welsch-transformed Pearson-correlation distance matrix.

    Parameters
    ----------
    stocks_returns:
        2-D array of shape (T, n).
    min_obs:
        Minimum observations required for a valid pairwise Pearson correlation.
        Pairs below this threshold are treated as uncorrelated (distance = 1.0).

    Returns
    -------
    np.ndarray of shape (n, n) with values in [0, 1].  Diagonal is 0.
    """
    welsch = lambda x: 1 - np.exp(-0.5 * x)
    distance_func = lambda di: np.sqrt(0.5 * (1 - di))

    corr_df = pd.DataFrame(stocks_returns).corr(min_periods=min_obs)
    corr_matrix = corr_df.to_numpy()
    corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
    corr_matrix = np.nan_to_num(corr_matrix, nan=-1.0, posinf=-1.0, neginf=-1.0)

    distance_matrix = distance_func(corr_matrix)
    distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=1.0, neginf=1.0)
    welsch_distance = np.nan_to_num(welsch(distance_matrix), nan=1.0, posinf=1.0, neginf=1.0)
    np.fill_diagonal(welsch_distance, 0.0)
    return np.clip(welsch_distance, 0.0, 1.0)
