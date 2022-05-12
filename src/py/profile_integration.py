from typing import Sequence
from sparseSpACE.Function import *
from sparseSpACE.StandardCombi import *
from sparseSpACE.Grid import *
from sparseSpACE.GridOperation import *
import numpy as np
from scipy.stats import norm
from option import AsianOption, EuropeanOption


class GeomAsianPayout(Function):
    def __init__(self, S_0=100, K=100, r=0.1, sigma=0.2, T=1.0, d=8):
        super().__init__()
        self.S_0 = S_0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.d = d
        self.t_v = np.linspace(0, 1, self.d, endpoint=False) + 1 / self.d
        self.delta_t = self.t_v[0]
        # random walk brownian
        self.A = np.sqrt(self.delta_t) * np.tril(np.ones(self.d))

        self.M = np.log(self.S_0)+0.5*(self.r-.5*self.sigma**2)*(self.T+self.delta_t)
        self.Aj = self.A.sum(axis=0)
        self.gamma_d = (self.d-np.arange(self.d)+2)*np.sqrt(self.T)*sigma/(d*np.sqrt(d))

    def brownian_bridge_generator(self):
        st = np.mgrid[1 : self.d + 1, 1 : self.d + 1] / self.d
        cov = st.min(axis=0) * self.sigma**2
        return np.linalg.cholesky(cov)

    def S_t(self, x: np.ndarray):
        return self.S_0 * np.exp(
            (self.r - 0.5 * self.sigma**2) * self.t_v
            + self.sigma * self.A@(norm.ppf(x))
        )

    def payout_func(self, x: np.ndarray):
        """Geometric Asian payoff
        """
        payout = np.maximum(
            0, np.power(np.prod(x, axis=0), 1 / self.d) - self.K
        )
        return payout

    def eval(self, coordinates):
        """Payout for values x in [0,1]^d
        """
        payout = np.exp(-self.r*self.T+self.M) * np.maximum(
            0, np.exp(
                np.inner(self.gamma_d, norm.ppf(coordinates))-np.exp(-self.M)*self.K
            )
        )
        return payout

    def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
        """Payout for values x in [0,1]^d
        """
        payout = np.exp(-self.r*self.T+self.M) * np.maximum(
            0, np.exp(
                np.inner(self.gamma_d, norm.ppf(coordinates))-np.exp(-self.M)*self.K
            )
        )
        
        return payout

    # def eval_vectorized(self, coordinates: Sequence[Sequence[float]]):
    
    #     return np.apply_along_axis(
    #         lambda x: self.payout_func(self.S_t(x)),
    #         axis=1, arr=coordinates
    #     )
if __name__ == "__main__":
    aop = AsianOption(d=5)
    dim = aop.d
    aop_payout = lambda x: aop.payout_func(aop.S_t(x))
    #aop_payout = aop.payout_func_opt
    sparse_payout = CustomFunction(aop_payout)
    a = np.zeros(dim) + aop.epsilon
    b = np.ones(dim) - aop.epsilon
    
    grid = GaussLegendreGrid(a=a, b=b)
    f = GeomAsianPayout(d=aop.d)
    # NEW! define operation which shall be performed in the combination technique
    from sparseSpACE.GridOperation import *
    operation = Integration(f=f, grid=grid, dim=dim, reference_solution=EuropeanOption().scholes_call())
    combiObject = StandardCombi(a, b, operation=operation, n_jobs=16)
    minimum_level = 1
    maximum_level = 6
    combiObject.perform_operation(minimum_level, maximum_level, f)