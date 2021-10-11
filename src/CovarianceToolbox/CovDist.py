import numpy as np
import pandas as pd

def CovDist(A, method, power=1.0):
    """
    Compute Pairwise Distance for Symmetric Positive-Definite Matrices

    For a given 3-dimensional array where Symmetric Positive Definite (SPD)
    matrices are stacked slice by slive, it computes pairwise distance using
    various popular measures. Some of the measures are metric, as they suffice
    3 conditions in a mathematical context; nonnegative definiteness, symmetry,
    and triangle inequalities. Other non-metric measures represent dissimilarities
    between two SPD objects.

    params:
    
    A - a (p x p) 3d array of N SPD matrices.
    method - the type of distance measured to be used.
    power - a non-zero number for PowerEuclidean disstance.

    Returns an (N x N) symmetric matrix of pairwise distances.
    """

    # Preprocessing
    # 1) 3d-array 2) square 3) symmetric 4) sequentially check PDness

    try:
        if type(A) == list:
            A = np.array(A)
    except TypeError:
        pass # ?

    if len(A.shape) != 3:
        raise TypeError
        # Raise error saying we need 3d array
    
    if (A.shape[0] != A.shape[1]):
        raise TypeError
        # Raise error saying matrix A should be stack of square matrices

    # Symmetric check & PosDef Check
    nonSymmetric = []
    nonPosDef = []
    for i in range(len(A)):
        if not np.allclose(A[i], A[i].T):
            nonSymmetric.append(i)
        if not np.all(np.linalg.eigvals(A[i]) > 0):
            nonPosDef.append(i)

    if len(nonSymmetric) != 0:
        raise TypeError
        # Matrices at indexes of i are not symmetric

    if len(nonPosDef) != 0:
        raise TypeError
        #Matrices at indexes of i are not Positive Definite

    # TODO to perform individual calculations based on method param

    return A
    
