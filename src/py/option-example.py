from tqdm import tqdm
import numpy as np 
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from option import AsianOption
from tasmanian_tools import make_grid

# %%
s_0_l=80.0
s_0_r=120.0
sigma_l=0.1
sigma_r=0.2
mu_l=0.02
mu_r=0.05
T_l=0.9
T_r=1.0
K_l=109.0
K_r=110.0

# %%
# max possible dim, level before 16 GB RAM is full: 40, 4
dim = 16
aop = AsianOption(d=dim)
level = 4
lb, rb = np.zeros(dim), np.ones(dim)

grid = make_grid(dim, level, lb, rb, "gauss-patterson")
weights, points = grid.getQuadratureWeights(), grid.getPoints()
pre_factor = np.exp(-aop.r*aop.T+aop.M)
payout_v = aop.payout_func_opt(points)
pre_factor*np.inner(weights, payout_v), aop.scholes_call()

# %%
grid2d = make_grid(2, 2, np.zeros(2), np.ones(2), "gauss-patterson")
weights2d, points2d = grid2d.getQuadratureWeights(), grid2d.getPoints()
plt.scatter(points2d[:, 0], points2d[:, 1])
plt.title("Gauss-Patterson Sparse Grid")
#plt.savefig("Gauss_patterson_sg.PNG")


# %%
s_0 = np.linspace(s_0_l, s_0_r, 21)
sigma = np.linspace(sigma_l, sigma_r, 5)
mu = np.linspace(mu_l, mu_r, 9)
T = np.linspace(T_l, T_r, 9)
K = np.linspace(K_l, K_r, 9)

# %%
mesh = np.meshgrid(s_0, sigma, mu, T, K)
grid_ar = np.stack(mesh, -1).reshape(-1, 5)

## vectorizing training data generation takes up too much memory
# -> make a loop

#for s_0, sigma, mu, T, K in grid_ar[:10]:
#    print(s_0, sigma, mu, T, K)


# %%
#grid_ar.shape[0]
Y = np.empty((grid_ar.shape[0], 2), dtype=np.float64)
i = 0
for s_0, sigma, mu, T, K in tqdm(grid_ar):
    aop = AsianOption(S_0=s_0, sigma=sigma, r=mu, T=T, K=K, d=dim)
    pre_factor = np.exp(-aop.r*aop.T+aop.M)
    payout_v = aop.payout_func_opt(points)
    Y[i] = pre_factor*np.inner(weights, payout_v), aop.scholes_call()
    i=+1
    
np.save("exports/sparse_geom_asian.npy", Y)

# %%
ar = np.load("notebooks/sparse_geom_asian_out.npy")

# %%
#plt.scatter(range(10), np.abs(np.diff(ar[990:1000])))
op = AsianOption(
    S_0=grid_ar[999, 0],
    sigma=grid_ar[999, 1],
    r=grid_ar[999, 2], T=grid_ar[999, 3], K=grid_ar[999, 4], d=dim)
op.scholes_call()


