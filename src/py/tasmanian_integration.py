import math
from typing import Callable
import numpy as np
from joblib import Parallel, delayed
import Tasmanian
import matplotlib.pyplot as plt

from option import AsianOption, EuropeanOption
from profile_integration import GeomAsianPayout
from genz_quad_examples import IntegrationTestFunctions
from configuration.config import settings


def make_grid( dim, exactness, lb, rb, rule):
    """Creates a sparse grid according to given rules and boundaries.

    Args:
        dim (int): Input Dimension.
        exactness (int): Exactness of the integration.
        lb (array-like): Left boundary. Shape (dim) is recommended.
        rb (array-like): Right boundary. Shape (dim) is recommended.
        rule (str):  One of the local polynomial rules in TASMANIAN docs.
            Defaults to "localp".

    Returns:
        TasmanianSparseGrid: SparseGrid object.
    """
    grid = Tasmanian.makeGlobalGrid(
        dim, 1, exactness, "qptotal", rule
    )
    grid.setDomainTransform(np.vstack([lb, rb]).T)
    return grid

#@nb.jit
def compute_integral(points: np.ndarray, weights: np.ndarray, f: Callable):
    #f_p = np.empty_like(weights)
    f_p = np.apply_along_axis(f, axis=1, arr=points)
    return np.inner(weights, f_p)


def parallel_compute_integral(points: np.ndarray, weights: np.ndarray, f: Callable):
    arr_length = points.shape[0]
    results = Parallel(n_jobs=16)((f)(points[i, :]) for i in range(arr_length))
    return np.inner(weights, results)
    pass

if __name__ == "__main__":
    dim = 3
    exact = settings.results3d.continuous
    f = IntegrationTestFunctions().continuous_f
    lb, rb = np.zeros(dim), np.ones(dim)
    grid = make_grid(dim, 51, lb, rb, "gauss-patterson")
    weights, points = grid.getQuadratureWeights(), grid.getPoints()
    approx_integral = compute_integral(points, weights, f)
    print(
        f"Log10 Integrationsfehler: {np.log10(np.abs(approx_integral-exact)):.2f}"
    )
