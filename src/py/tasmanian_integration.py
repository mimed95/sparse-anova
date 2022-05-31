import math
from typing import Callable
import numpy as np
import numba as nb
import Tasmanian
import matplotlib.pyplot as plt

from option import AsianOption, EuropeanOption
from profile_integration import GeomAsianPayout
from genz_quad_examples import IntegrationTestFunctions
from configuration.config import settings


def make_grid( dim, depth, lb, rb, rule):
    """Creates a sparse grid according to given rules and boundaries.

    Args:
        dim (int): Input Dimension.
        depth (int): Depth of the hierachisation tree.
        lb (array-like): Left boundary. Shape (dim) is recommended.
        rb (array-like): Right boundary. Shape (dim) is recommended.
        rule (str):  One of the local polynomial rules in TASMANIAN docs.
            Defaults to "localp".

    Returns:
        TasmanianSparseGrid: SparseGrid object.
    """
    grid = Tasmanian.makeGlobalGrid(
        dim, 1, depth, "qptotal", rule
    )
    grid.setDomainTransform(np.vstack([lb, rb]).T)
    return grid


def integral_error(grid, f, fExact):
    aPoints = grid.getPoints()
    aWeights = grid.getQuadratureWeights()
    # fApproximateIntegral = np.sum(aWeights * np.apply_along_axis(f, axis=0, arr=aPoints))
    # fError = np.abs(fApproximateIntegral - fExact)
    # return "{0:>10d}{1:>10.2e}".format(grid.getNumPoints(), fError)
    return aWeights, aPoints

#@nb.jit
def compute_integral(points: np.ndarray, weights: np.ndarray, f: Callable):
    f_p = np.empty_like(weights)
    f_p = np.apply_along_axis(f, axis=1, arr=points)
    return np.inner(weights, f_p)

if __name__ == "__main__":
    dim = 2
    exact = settings.results2d.continuous
    f = IntegrationTestFunctions().continuous_f
    lb, rb = np.zeros(dim), np.ones(dim)

    grid = make_grid(dim, 51, lb, rb, "gauss-patterson")
    weights, points = integral_error(grid, f, exact)
    #Tasmanian.loadNeededValues(lambda x, tid: np.ones((1,))*np.exp(-x), grid, 4)
    print(
        compute_integral(points, weights, f),
        exact
    )
    # print(weights.shape, points.shape)
    # print("               Clenshaw-Curtis      Gauss-Legendre    Gauss-Patterson")
    # print(" precision    points     error    points     error    points    error")
    # grid.plotResponse2D()
    # plt.savefig("/mnt/c/Users/Michael/Pictures/polyresponse.PNG")

    # for prec in range(5, 41, 5):
    #     print("{0:>10d}{1:1s}{2:1s}{3:1s}".format(
    #         prec,
    #         integral_error(
    #             make_grid(prec, "clenshaw-curtis", dim), lambda x: f(x), exact
    #         ),
    #         integral_error(
    #             make_grid(prec, "gauss-legendre-odd", dim), lambda x: f(x), exact
    #         ),
    #         integral_error(
    #             make_grid(prec, "gauss-patterson", dim), lambda x: f(x), exact)
    #         )
    #     )