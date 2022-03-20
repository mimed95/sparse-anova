import pysgpp
# the standard parabola (arbitrary-dimensional)


def f(x):
  res = 1.0
  for i in range(len(x)):
      res *= 4.0*x[i]*(1.0-x[i])
  return res
# a pyramid-like shape (arbitrary-dimensional)
def g(x):
  res = 1.0
  for i in range(len(x)):
      res *= 2.0*min(x[i], 1.0-x[i])
  return res

if __name__ == '__main__':
    dim = 2
    grid = pysgpp.Grid.createLinearGrid(dim)
    gridStorage = grid.getStorage()
    print("dimensionality:        {}".format(dim))
    # create regular grid, level 3
    level = 3
    gridGen = grid.getGenerator()
    gridGen.regular(level)
    print("number of grid points: {}".format(gridStorage.getSize()))

    # create coefficient vector
    alpha = pysgpp.DataVector(gridStorage.getSize())
    for i in range(gridStorage.getSize()):
      gp = gridStorage.getPoint(i)
      p = tuple([gp.getStandardCoordinate(j) for j in range(dim)])
      alpha[i] = f(p)
    pysgpp.createOperationHierarchisation(grid).doHierarchisation(alpha)

    # direct quadrature
    opQ = pysgpp.createOperationQuadrature(grid)
    res = opQ.doQuadrature(alpha)
    print("exact integral value:  {}".format(res))
    # Monte Carlo quadrature using 100000 paths
    opMC = pysgpp.OperationQuadratureMC(grid, 100000)
    res = opMC.doQuadrature(alpha)
    print("Monte Carlo value:     {:.6f}".format(res))
    res = opMC.doQuadrature(alpha)
    print("Monte Carlo value:     {:.6f}".format(res))
    # Monte Carlo quadrature of a standard parabola
    res = opMC.doQuadratureFunc(f)
    print("MC value (f):          {:.6f}".format(res))
    # Monte Carlo quadrature of error
    res = opMC.doQuadratureL2Error(f, alpha)
    print("MC L2-error (f-u)      {:.7f}".format(res))
    # Monte Carlo quadrature of a piramidal function
    res = opMC.doQuadratureFunc(g)
    print( "MC value (g):          {:.6f}".format(res))
    # Monte Carlo quadrature of error
    res = opMC.doQuadratureL2Error(g, alpha)
    print( "MC L2-error (g-u)      {:.7f}".format(res))
    