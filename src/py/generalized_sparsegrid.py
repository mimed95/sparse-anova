import numpy as np
import pysgpp as sg

sg.omp_set_num_threads(10)
# import pandas as pd
import sklearn.datasets as data


def generate_friedman1(seed):
    (X, y) = data.make_friedman1(n_samples=10000, random_state=seed, noise=1.0)
    y = sg.DataVector(y)
    X = sg.DataMatrix(X)
    return X, y

# The grid granularities are controlled by the parameter 
def evaluate(X_tr, y_tr, X_te, y_te, T):
    grid = sg.RegularGridConfiguration()
    grid.dim_ = 10
    grid.level_ = 4
    grid.t_ = T
    grid.type_ = sg.GridType_ModLinear
    adapt = sg.AdaptivityConfiguration()
    adapt.numRefinements_ = 5
    adapt.noPoints_ = 3
    solv = sg.SLESolverConfiguration()
    solv.maxIterations_ = 50
    solv.eps_ = 10e-6
    solv.threshold_ = 10e-6
    solv.type_ = sg.SLESolverType_CG
    final_solv = solv
    final_solv.maxIterations = 200
    regular = sg.RegularizationConfiguration()
    regular.type_ = sg.RegularizationType_Identity
    regular.exponentBase_ = 1.0
    regular.lambda_ = 10e-4
    estimator = sg.RegressionLearner(grid, adapt, solv, final_solv, regular)
    estimator.train(X_tr, y_tr)
    print(estimator.getGridSize())
    return estimator.getMSE(X_te, y_te)


def main():
    X_tr, y_tr = generate_friedman1(123)
    X_te, y_te = generate_friedman1(345)
    Ts = [-0.5, 0, 0.5, 1.0]
    for T in Ts:
        mse = evaluate(X_tr, y_tr, X_te, y_te, T)
        print(
            "The sparse grid with T={:2.1f} achieved a testing RMSE of {:2.4f}.".format(
                T, np.sqrt(mse)
            )
        )


if __name__ == "__main__":
    main()
