import Tasmanian
import numpy as np


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