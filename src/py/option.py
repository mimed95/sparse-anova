from dataclasses import dataclass
from re import T
import numpy as np
import chaospy as cp
from numba import jit
from scipy.stats import norm


@dataclass()
class Option:
    S_0: float = 100
    K: float = 100
    r: float = 0.1
    T: float = 1
    sigma: float = 0.2
    epsilon: float = 1e-10
    d: int = 16

    def __post_init__(self):
        self.t_v = np.linspace(0, 1, self.d, endpoint=False) + 1 / self.d
        self.delta_t = self.t_v[0]
        # random walk brownian
        self.A = np.sqrt(self.delta_t) * np.tril(np.ones(self.d))

    def brownian_bridge_generator(self):
        st = np.mgrid[1 : self.d + 1, 1 : self.d + 1] / self.d
        cov = st.min(axis=0) * self.sigma**2
        return np.linalg.cholesky(cov)


@dataclass
class EuropeanOption(Option):
    S_0: float = 100
    K: float = 100
    r: float = 0.1
    sigma: float = 0.2
    epsilon: float = 1e-10
    d: int = 16

    def S_t(self, x: np.ndarray):
        """Calculate the asset price in the Black-Scholes model.

        Args:
            x (np.ndarray): Array with values in (0,1)

        Returns:
            np.ndarray: Asset price evolution.
        """
        S = self.S_0 * np.exp(
            (self.r - 0.5 * self.sigma**2) * self.T
            + self.sigma * np.sqrt(self.T)*(norm.ppf(x))
        )
        return S

    def payout_func(self):
        payout = np.exp(-self.r*self.T)*np.maximum(0, self.S_t(self.T) - self.K)
        return payout

    def scholes_call(self, St=None, t=0):
        if St is None and t==0:
            St=self.S_0
        tt_maturity = T-t
        d1 = 1/(self.sigma*np.sqrt(tt_maturity))*(
            np.log(St/self.K)+(self.r+0.5*self.sigma**2)*tt_maturity
        )
        d2 = d1-self.sigma*np.sqrt(tt_maturity)
        return norm.cdf(d1)*St-norm.cdf(d2)*self.K*np.exp(-self.r*tt_maturity)

@dataclass()
class AsianOption(Option):
    S_0: float = 100
    K: float = 100
    r: float = 0.1
    sigma: float = 0.2
    T: float = 1
    d: int = 16
    level: int = 3
    epsilon: float = 1e-10

    def __post_init__(self):
        self.t_v = np.linspace(0, 1, self.d, endpoint=False) + 1 / self.d
        self.delta_t = self.t_v[0]
        # random walk brownian
        self.A = np.sqrt(self.delta_t) * np.tril(np.ones(self.d))
    
    def S_t(self, x: np.ndarray):
        return self.S_0 * np.exp(
            (self.r - 0.5 * self.sigma**2) * self.t_v
            + self.sigma * self.A@(norm.ppf(x))
        )

    def payout_func(self, x: np.ndarray):
        """Geometric Asian payoff
        """
        payout = np.maximum(
            0, np.prod(x, axis=0)** (1 / self.d) - self.K
        )
        return payout

    def gen_quad(self, lb=0, ub=1, rule=None):
        """Quadrature node and weight generator.

        Args:
            lb (int, optional): [description]. Defaults to 0.
            ub (int, optional): [description]. Defaults to 1.
            rule (str, optional): Name of the quadrature rule.
                For rules refer to the chaospy.quadrature module.
                Defaults to None.

        Returns:
        (numpy.ndarray, numpy.ndarray):
            Abscissas and weights created from full tensor grid rule. Flatten
            such that ``abscissas.shape == (len(dist), len(weights))``.
        """
        iid_dist = cp.Iid(cp.Uniform(lb + self.epsilon, ub - self.epsilon), self.d)
        return cp.generate_quadrature(self.level, iid_dist, sparse=True, rule=rule)

    def compute_premium(self):
        abscissas, weights = self.gen_quad()
        payout_v = self.payout_func(abscissas)
        return np.dot(weights, payout_v) * np.exp(-self.r * self.T)
    
    def scholes_call(self, t=0):
        t1 = self.T - (
            self.d*(self.d-1)*(4*self.d+1)*self.delta_t
        ) / (6*self.d**2)
        t2 = self.T-0.5*(self.d-1)*self.delta_t
        beta = (
            np.log(self.S_0/self.K)+(self.r-0.5*self.sigma**2)*t2
        ) / (self.sigma*np.sqrt(t1))
        gamma = np.exp(
            -self.r*(self.T-t2)-0.5*self.sigma**2*(t2-t1)
        )
        return (
            self.S_0*gamma*norm.cdf(beta+self.sigma*np.sqrt(t1)) -
            self.K*np.exp(-self.r*self.T)*norm.cdf(beta)
        )
        

def sde_body(idx, s, sigma,mu,T,K, samples, batch_size, N=128): 
    h=T/N
    z=np.random_normal(shape=(samples, batch_size,1),
                          stddev=1., dtype=np.float32)
    s=s + mu *s * h +sigma * s *np.sqrt(h)*z + 0.5 *sigma *s *sigma * ((np.sqrt(h)*z)**2-h)    
    return s