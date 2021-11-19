import math
from pysgpp import *
import matplotlib.pyplot as plotter
from mpl_toolkits.mplot3d import Axes3D


def calculateError(dataSet,f,grid,alpha,error):
    print("calculating error")
    #traverse dataSet
    vec = DataVector(2)
    opEval = createOperationEval(grid)
    for i in range(dataSet.getNrows()):
        dataSet.getRow(i,vec)
        error[i] = pow(
            f(dataSet.get(i,0),dataSet.get(i,1))-opEval.eval(alpha,vec),
            2)
    return error

if __name__ == '__main__':
    f = lambda x0, x1: math.sin(x0*10)+x1
    dim = 2
    grid = Grid.createLinearGrid(dim)
    HashGridStorage = grid.getStorage()
    print("dimensionality:                   {}".format(dim))
    # create regular grid, level 3
    level = 3
    gridGen = grid.getGenerator()
    gridGen.regular(level)
    print("number of initial grid points:    {}".format(HashGridStorage.getSize()))
    # create coefficient vectors
    alpha = DataVector(HashGridStorage.getSize())
    print("length of alpha vector:           {}".format(alpha.getSize()))
    
    rows = 100
    cols = 100
    dataSet = DataMatrix(rows*cols,dim)
    vals = DataVector(rows*cols)
    for i in range(rows):
        for j in range(cols):
            #xcoord
            dataSet.set(i*cols+j,0,i*1.0/rows)
            #ycoord
            dataSet.set(i*cols+j,1,j*1.0/cols)
            vals[i*cols+j] = f(i*1.0/rows,j*1.0/cols)

# create coefficient vectors
    alpha = DataVector(HashGridStorage.getSize())
    print("length of alpha vector:           {}".format(alpha.getSize()))
    # now refine adaptively 20 times
    for refnum in range(20):
            for i in range(HashGridStorage.getSize()):
                gp = HashGridStorage.getPoint(i)
                alpha[i] = f(gp.getStandardCoordinate(0), gp.getStandardCoordinate(1))
            # hierarchize
            createOperationHierarchisation(grid).doHierarchisation(alpha)
    
            errorVector = DataVector(dataSet.getNrows())
            calculateError(dataSet, f, grid, alpha, errorVector)
                #refinement  stuff
            refinement = ANOVAHashRefinement()
            decorator = PredictiveRefinement(refinement)
            # refine a single grid point each time
            print(f"Error over all = {errorVector.sum():.3f}" )
            indicator = PredictiveRefinementIndicator(
                grid,dataSet,errorVector,1)
            decorator.free_refine(HashGridStorage,indicator)
            print("Refinement step %d, new grid size: %d" % (refnum+1, HashGridStorage.getSize()))
            # extend alpha vector (new entries uninitialized)
            alpha.resizeZero(HashGridStorage.getSize())
