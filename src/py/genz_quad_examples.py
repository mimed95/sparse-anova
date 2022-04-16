import logging
from typing import Callable, Iterable

from configuration.config import settings
import numpy as np
import pysgpp as sg


FORMAT = '%(asctime)s :: %(levelname)s :: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

# For reference see the testfunctions at https://www.sfu.ca/~ssurjano/cont.html
class IntegrationTestFunctions:
    def __init__(self, u=0.5, a=5):
        self.u = u
        self.a = a

    def cont_f(self, x):
        """
        using u = (0.5,..,0.5), a_i=5
        """
        return np.exp(
            -self.a*np.sum([abs(x[i]-self.u) for i in range(len(x))])
        )

    def cornerpeak_f(self, x):
        """
        using u = (0.5,..,0.5), a_i=5
        """
        return (
            1+self.a*np.sum([x[i] for i in range(len(x))])
        )**-(len(x)+1)

    def gausspeak_f(self, x):
        """
        using u = (0.5,..,0.5), a_i=5
        """
        return np.exp(
            -self.a**2*np.sum([(x[i]-self.u)**2 for i in range(len(x))])
        )

    def oscillatory_f(self, x):
        """
        using u = (0.5,..,0.5), a_i=5
        """
        if isinstance(self.u, Iterable):
            u1 = self.u[0]
        else:
            u1 = self.u
        return np.cos(u1*2*np.pi+self.a*np.sum([x[i] for i in range(len(x))]))

    def productpeak_f(self, x):
        """
        using u = (0.5,..,0.5), a_i=5
        """
        return np.prod(
            [1/(self.a**(-2)+(x[i]-self.u)**2)] for i in range(len(x))
        )
    
    def discont_f(self, x):
        """
        using u = (0.5,..,0.5), a_i=5
        """
        if x[0] > self.u or x[1] > self.u:
            return 0
        return np.exp(
            self.a*np.sum([x[i] for i in range(len(x))])
        )


def hierarchise(func: Callable, dim: int, level: int):
    grid = sg.Grid.createLinearGrid(dim)
    gridStorage = grid.getStorage()
    logging.info("dimensionality:        {}".format(dim))
    logging.info(f"Level: {level}")
    gridGen = grid.getGenerator()
    gridGen.regular(level)
    logging.info("number of grid points: {}".format(gridStorage.getSize()))
    # create coefficient vector
    alpha = sg.DataVector(gridStorage.getSize())
    for i in range(gridStorage.getSize()):
        gp = gridStorage.getPoint(i)
        p = np.array([gp.getStandardCoordinate(j) for j in range(dim)])
        alpha[i] = func(p)
    sg.createOperationHierarchisation(grid).doHierarchisation(alpha)
    return grid, alpha


if __name__ == "__main__":
    Fs = IntegrationTestFunctions()
    func = Fs.cont_f
    dim = 3
    level = 5
    grid, alpha = hierarchise(func, dim, level)
    # direct quadrature
    opQ = sg.createOperationQuadrature(grid)
    sg_res = opQ.doQuadrature(alpha)

    opMC = sg.OperationQuadratureMC(grid, 100000)
    res = opMC.doQuadrature(alpha)
    logging.info("Monte Carlo value:     {:.6f}".format(res))
    # Monte Carlo quadrature of a standard parabola
    res = opMC.doQuadratureFunc(func)
    logging.info("MC value (f):          {:.6f}".format(res))
    # Monte Carlo quadrature of error
    res = opMC.doQuadratureL2Error(func, alpha)
    logging.info("MC L2-error (f-u)      {:.7f}".format(res))
    if dim == 2:
        exact_resf1 = settings.results2d.continous # in d=2
    elif dim == 3:
        exact_resf1 = settings.results3d.continous # in d=3
    else:
        pass
    logging.info(f"integral value:  {sg_res:.6f}")
    logging.info(f"exact integral value: {exact_resf1:.6f}")
    logging.info(f"Log10 error: {-np.log10(np.abs(sg_res-exact_resf1)):.2f}")