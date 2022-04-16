# Copyright (C) 2008-today The SG++ project
# This file is part of the SG++ project. For conditions of distribution and
# use, please see the copyright notice provided with SG++ or at
# sgpp.sparsegrids.org

# --------------------------------------------------
# Quadrature test setting
# --------------------------------------------------
from math import exp, sin
from sympy import Symbol, integrate
from sympy import exp as exponential
from sympy import sin as sinus
import unittest

from pysgpp import *


def interpolate(f, level, dim, gridType=GridType_Linear, deg=2, trans=None):
    # create a two-dimensional piecewise bi-linear grid
    if gridType == GridType_PolyBoundary:
        grid = Grid.createPolyBoundaryGrid(dim, deg)
    elif gridType == GridType_Poly:
        grid = Grid.createPolyGrid(dim, deg)
    elif gridType == GridType_Linear:
        grid = Grid.createLinearGrid(dim)
    elif gridType == GridType_LinearBoundary:
        grid = Grid.createLinearBoundaryGrid(dim, 1)
    else:
        raise AttributeError

    gridStorage = grid.getStorage()

    # create regular grid
    grid.getGenerator().regular(level)

    # create coefficient vector
    alpha = DataVector(gridStorage.getSize())
    alpha.setAll(0.0)

    # set function values in alpha
    x = DataVector(dim)
    for i in range(gridStorage.getSize()):
        gp = gridStorage.getPoint(i)
        gridStorage.getCoordinates(gp, x)
        p = x.array()

        if trans is not None:
            p = trans.unitToProbabilistic(p)

        if gridStorage.getDimension() == 1:
            p = p[0]
        alpha[i] = f(p)

    # hierarchize
    createOperationHierarchisation(grid).doHierarchisation(alpha)

    return grid, alpha


class QuadratureTest(unittest.TestCase):

    def testQuadratureSG(self):
        symx = Symbol('x', real=True)

        def quad(f, g):
            intf = integrate(g, symx)
            grid, alpha = interpolate(f, 4, 1)

            f1 = createOperationQuadrature(grid).doQuadrature(alpha)
            f2 = intf.subs(symx, 1) - intf.subs(symx, 0)

            return f1, f2.evalf()

        tests = [(lambda x: 6. * x * (1. - x), 6 * symx * (1 - symx)),
                 (lambda x: x ** 3 - x ** 2, symx ** 3 - symx ** 2)]
#                  (exp, exponential(symx)),
#                  (lambda x: sin(x) + x, sinus(symx) + symx)]

        for i, (f, g) in enumerate(tests):
            f1, f2 = quad(f, g)
            self.assertTrue((abs(f1 - f2) / f1) < 1e-2,
                            "%i: |%g - %g| / %g = %g >= 1e-2" % (i, f1, f2, f1, (abs(f1 - f2) / f1)))

    def testQuadratureTruncated(self):
        def f(x): return 1.
        grid, alpha = interpolate(f, 2, 3)
        alpha = DataVector(grid.getStorage().getSize())

        for ix in range(0, grid.getStorage().getSize()):
            alpha.setAll(0.0)
            alpha[ix] = 1.
            gp = grid.getStorage().getPoint(ix)

            accLevel = sum([max(1, gp.getLevel(d)) for d in range(gp.getDimension())])
            self.assertTrue(createOperationQuadrature(grid).doQuadrature(alpha) == 2 ** -accLevel,
                            "%g != %g" % (createOperationQuadrature(grid).doQuadrature(alpha), 2 ** -accLevel))

    def testQuadraturePolynomial(self):
        symx = Symbol('x', real=True)
        symy = Symbol('y', real=True)

        def quad(f, g, dim):
            intf = integrate(g, (symx, 0, 1))
            if dim == 2:
                intf = integrate(intf, (symy, 0, 1))

            grid, alpha = interpolate(f, 4, dim, GridType_Linear)
            grid2, alpha2 = interpolate(f, 4, dim, GridType_PolyBoundary, 2)

            f1 = createOperationQuadrature(grid).doQuadrature(alpha)
            f2 = createOperationQuadrature(grid2).doQuadrature(alpha2)
            f3 = intf.evalf()

            return f1, f2, f3

        tests = [(lambda x: 6. * x[0] ** 3 * (1. - x[1] ** 2), 6 * symx ** 3 * (1 - symy ** 2), 2),
                 (lambda x: x ** 3 - x ** 2, symx ** 3 - symx ** 2, 1),
                 (lambda x: exp(x), exponential(symx), 1),
                 (lambda x: sin(x) + x, sinus(symx) + symx, 1)]

        for f, g, dim in tests:
            f1, f2, f3 = quad(f, g, dim)
            assert abs(f3 - f1) >= abs(f3 - f2)

# --------------------------------------------------
# testing
# --------------------------------------------------


# if __name__ == "__main__":
#     unittest.main()