import logging
from typing import Callable
import numpy as np
from joblib import Parallel
import Tasmanian
import matplotlib.pyplot as plt

from option import AsianOption, EuropeanOption
from profile_integration import GeomAsianPayout
from genz_quad_examples import IntegrationTestFunctions
from configuration.config import settings

FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)


def make_grid(dim, level, lb, rb, rule):
    """Creates a sparse grid according to given rules and boundaries.

    Args:
        dim (int): Input Dimension.
        level (int): level of the integration.
        lb (array-like): Left boundary. Shape (dim) is recommended.
        rb (array-like): Right boundary. Shape (dim) is recommended.
        rule (str):  One of the local polynomial rules in TASMANIAN docs.
            Defaults to "localp".

    Returns:
        TasmanianSparseGrid: SparseGrid object.
    """
    grid = Tasmanian.makeGlobalGrid(
        dim, 1, level, "level", rule
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


if __name__ == "__main__":
    dim = 4
    level = 7
    #exact = settings.results3d.discontinuous
    #f = IntegrationTestFunctions().discontinuous_f
    aop = AsianOption(d=dim)
    f = lambda x: aop.payout_func_opt(x)
    exact = aop.scholes_call()
    lb, rb = np.zeros(dim), np.ones(dim)
    grid = make_grid(dim, level, lb, rb, "gauss-patterson")
    weights, points = grid.getQuadratureWeights(), grid.getPoints()
    res = compute_integral(points, weights, f)
    print(res, exact)
    print(
        f"Log10 Integrationsfehler: {np.log10(np.abs(res-exact)):.2f}"
    )
    # Fs = IntegrationTestFunctions()
    # for func_name in settings.results2d.keys():
    #     logging.info(f"Sparse Grid Integration for {func_name}")
    #     func = getattr(Fs, func_name +"_f")
    #     weights, points = grid.getQuadratureWeights(), grid.getPoints()
    #     approx_integral = compute_integral(points, weights, func)

    #     if dim == 2:
    #         exact_resf1 = settings.results2d[func] # in d=2
    #     elif dim == 3:
    #         exact_resf1 = settings.results3d[func] # in d=3
    #     else:
    #         pass
    #     logging.info(f"integral value:  {approx_integral:.6f}")
    #     logging.info(f"exact integral value: {exact_resf1:.6f}")

