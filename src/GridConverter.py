import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from pysgpp import multiFunc, DataVector, \
    CombigridMultiOperation, DataMatrix, SurplusRefinementFunctor, \
    Grid, convertCombigridToHierarchicalSparseGrid, convertHierarchicalSparseGridToCombigrid, \
    GridConversionTypes_ALLSUBSPACES, GridConversionTypes_COMPLETESUBSPACES, \
    createOperationHierarchisation, createOperationMultipleEval

def f(x):
    return np.prod([4 * xi * (1 - xi) for xi in x.array()])


# Spatially adaptive sparse grids
# to regular combination technique and back
def interpolate(grid, f):
    """
    This helper functions cmoputes the coefficients of a sparse grid
    function for a given function
    Arguments:
    grid -- Grid sparse grid from pysgpp
    f -- function to be interpolated
    Return DataVector coefficients of the sparse grid function
    """
    gs = grid.getStorage()
    alpha = DataVector(gs.getSize())
    p = DataVector(gs.getDimension())
    for i in range(gs.getSize()):
        gs.getCoordinates(gs.getPoint(i), p)
        alpha[i] = f(p)
    createOperationHierarchisation(grid).doHierarchisation(alpha)
    return alpha


def refineGrid(grid, alpha, f, refnums):
    """
    This function refines a sparse grid function refnum times.
    Arguments:
    grid -- Grid sparse grid from pysgpp
    alpha -- DataVector coefficient vector
    f -- function to be interpolated
    refnums -- int number of refinement steps
    Return nothing
    """
    gs = grid.getStorage()
    gridGen = grid.getGenerator()
    x = DataVector(gs.getDimension())
    for _ in range(refnums):
        # refine a single grid point each time
        gridGen.refine(SurplusRefinementFunctor(alpha, 1))
        # extend alpha vector (new entries uninitialized)
        alpha.resizeZero(gs.getSize())
        # set function values in alpha
        for i in range(gs.getSize()):
            gs.getCoordinates(gs.getPoint(i), x)
            alpha[i] = f(x)
        # hierarchize
        createOperationHierarchisation(grid).doHierarchisation(alpha)


def regularGridToRegularGrid(numDims,
                             level,
                             f,
                             numSamples=1000,
                             plot=False,
                             verbose=False):
    """
    Converts a regular sparse grid function to a sparse grid in the
    combination technique and back.
    Arguments:
    numDims -- int number of dimensions
    level -- level of the sparse grid
    f -- function to be interpolated
    numSamples -- int number of random samples on which we evaluate the different sparse grid
                  functions to validate the grid conversion
    plot -- bool whether the sparse grid functions are plotted or not (just for numDims=1)
    verbose -- bool verbosity
    """
    x = np.random.rand(numSamples, numDims)
    parameters = DataMatrix(x)
    grid = Grid.createLinearGrid(numDims)
    grid.getGenerator().regular(level)
    alpha = interpolate(grid, f)
    treeStorage_all = convertHierarchicalSparseGridToCombigrid(
        grid.getStorage(), GridConversionTypes_ALLSUBSPACES)
    treeStorage_complete = convertHierarchicalSparseGridToCombigrid(
        grid.getStorage(), GridConversionTypes_COMPLETESUBSPACES)
    func = multiFunc(f)
    opt_all = CombigridMultiOperation.createExpUniformLinearInterpolation(numDims, func)
    opt_complete = CombigridMultiOperation.createExpUniformLinearInterpolation(numDims, func)
    parameters.transpose()
    opt_all.setParameters(parameters)
    opt_all.getLevelManager().addLevelsFromStructure(treeStorage_all)
    opt_complete.setParameters(parameters)
    opt_complete.getLevelManager().addLevelsFromStructure(treeStorage_complete)
    parameters.transpose()
    if verbose:
        print("-" * 80)
        print("just full levels:")
        print(opt_complete.getLevelManager().getSerializedLevelStructure())
        print("-" * 80)
        print("all levels:")
        print(opt_all.getLevelManager().getSerializedLevelStructure())
        print("-" * 80)
    
    grid_complete = Grid.createLinearGrid(numDims)
    treeStorage_complete = opt_complete.getLevelManager().getLevelStructure()
    convertCombigridToHierarchicalSparseGrid(treeStorage_complete, grid_complete.getStorage())

    grid_all = Grid.createLinearGrid(numDims)
    treeStorage_all = opt_all.getLevelManager().getLevelStructure()
    convertCombigridToHierarchicalSparseGrid(treeStorage_all, grid_all.getStorage())

    alpha_complete = interpolate(grid_complete, f)
    alpha_all = interpolate(grid_all, f)
    y_sg_regular = DataVector(numSamples)
    createOperationMultipleEval(grid, parameters).eval(alpha, y_sg_regular)
    y_sg_all = DataVector(numSamples)
    createOperationMultipleEval(grid_all, parameters).eval(alpha_all, y_sg_all)
    y_sg_complete = DataVector(numSamples)
    createOperationMultipleEval(grid_complete, parameters).eval(alpha_complete, y_sg_complete)
    y_ct_all = opt_all.getResult()
    y_ct_complete = opt_complete.getResult()

    y_sg_regular = y_sg_regular.array().flatten()
    y_ct_all = y_ct_all.array().flatten()
    y_ct_complete = y_ct_complete.array().flatten()
    y_sg_all = y_sg_all.array().flatten()
    y_sg_complete = y_sg_complete.array().flatten()

    if plot and numDims == 1:
        x = x.flatten()
        ixs = np.argsort(x)
        plt.figure()
        plt.plot(x[ixs], y_sg_regular[ixs], label="sg regular")
        plt.plot(x[ixs], y_sg_all[ixs], label="sg all")
        plt.plot(x[ixs], y_ct_complete[ixs], label="ct full")
        plt.plot(x[ixs], y_ct_all[ixs], label="ct all")
        plt.legend()
        plt.show()
# running checks
    assert np.sum((y_ct_complete - y_ct_all) ** 2) < 1e-14
    assert np.sum((y_ct_complete - y_sg_regular) ** 2) < 1e-14
    assert np.sum((y_sg_regular - y_sg_all) ** 2) < 1e-14
    assert np.sum((y_sg_regular - y_sg_complete) ** 2) < 1e-14

    assert grid_complete.getSize() == grid.getSize()
    assert grid_all.getSize() == grid.getSize()


if __name__ == '__main__':
    parser = ArgumentParser(description='Get a program and run it with input')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument('--numDims', default=4, type=int,
                        help='number of dimensions')
    parser.add_argument('--level', default=4, type=int,
                        help='sparse grid level')
    parser.add_argument('--refnums', default=0, type=int,
                        help='number of refinement steps')
    parser.add_argument('--plot', default=False, action='store_true',
                        help='plot stuff')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='verbosity')
    args = parser.parse_args()
    # select the right conversion method based on the input parameters
    if args.refnums == 0:
        regularGridToRegularGrid(args.numDims,
                                 args.level,
                                 f,
                                 plot=args.plot,
                                 verbose=args.verbose)
#    else:
#        adaptiveGridToRegularGrid(args.numDims,
#                                  args.level,
#                                  args.refnums,
#                                  f,
#                                  plot=args.plot,
#                                  verbose=args.verbose)

#details on https://sgpp.sparsegrids.org/docs/example_gridConverter_py.html