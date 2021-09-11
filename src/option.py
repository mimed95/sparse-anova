from dataclasses import dataclass
import numpy as np
import chaospy as cp
from scipy.stats import norm


@dataclass()
class AsianOption:
    
    S_0: float = 100
    K: float = 100
    r: float = 0.1
    sigma: float = 0.2
    T: float = 1
    d: int = 16
    level: int = 3
    epsilon: float = 1e-10

    def __post_init__(self):
        self.t_v = np.linspace(0, 1, self.d, endpoint=False) + 1/self.d
        self.delta_t = self.t_v[0]
        # random walk brownian
        self.A = np.sqrt(self.delta_t)* np.tril(np.ones(self.d)) 

    def payout_func(self, x: np.ndarray):
        payout = np.maximum(0, 
            (1/self.d)*np.sum(
                self.S_0*np.exp(
                    (self.r+0.5*self.sigma**2)*self.t_v.reshape(self.d,1)
                    + self.sigma*self.A.dot(norm.ppf(x)) 
                ) - self.K, axis=0
            )
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
        iid_dist = cp.Iid(
            cp.Uniform(lb+self.epsilon, ub-self.epsilon), self.d
        ) 
        return cp.generate_quadrature(
            self.level,
            iid_dist,
            sparse=True,
            rule=rule
        )

    def compute_premium(self):
        abscissas, weights = self.gen_quad()
        payout_v = self.payout_func(abscissas)
        return np.dot(weights, payout_v)*np.exp(-self.r*self.T)