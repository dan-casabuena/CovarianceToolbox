###################################################
#                                                 #
#   Covariance Estimation via Hard Thresholding   #
#           Bickel and Levina (2008)              #
#        Implemented by Daniel Casabuena          #
#                                                 #
###################################################

from typing import Union, List
import numpy as np
import pandas as pd
import pytest

def estimate(features: Union[np.ndarray, pd.DataFrame, List[List]], threshold: Union[int, float] = None, nCV=10, parallel=False):
    """
    A function that performs Covariance Estimation via Hard Thresholding
    proposed by Bickel and Levina in their 2008 paper.

    The function requires a matrix of features, and an optional threshold
    value or set/array of values, which will be asserted using a cross validation
    technique also proposed within Bickel and Levina's 2008 paper.

    The technique is recommended for features that are sufficiently sparse enough
    with significantly more parameters than there is in the sample size.

    Returns an array; a thresholded sample covariance matrix.

    TODO Implement parallelization
    TODO Add check for positive definiteness
    """

    try:
        if type(features) == list:
            features = np.array(features)
    except TypeError:
        pass
        
    if threshold is None:
        threshold = np.sqrt(np.log(features.shape[0]) / features.shape[1])

    # Check if it is an iterable object; leads into cross validation method
    try:
        iter(threshold)
        return transform(features, _cross_validation(features, threshold, nCV))
    except TypeError:
        pass

    return transform(features, threshold)

def transform(X, threshold):
    """
    Calculates the empirical covariance matrix and performs hard thresholding.
    Pre-requisits must be satisfied in order to produce satisfactory results.

    Floors all off-diagonal values within the empirical covariance matrix to zero
    provided that: |S_i_j| <= t, where t is the threshold value.

    Returns the thresholded covariance matrix.
    """

    cov = np.cov(X, rowvar=False, bias=True)

    for i in range(len(cov)):
        for j in range(len(cov[i])):
            if abs(cov[i][j]) < threshold and i!=j:
                cov[i][j] = 0

    #TODO Check for errors at the beginning of the function

    return cov

def _cross_validation(X, param, nCV):
    """
    A cross validation method proposed in Bickel and Levina (2008) '3. Choice of threshold'

    Splits the sample into two 'justified' sizes, repeated N times. Utilizing the Frobenius metric
    and the respective empirical covariancce matrices based on the two split samples, the function
    finds the minimum cross-validation value from the function provided within the paper, provided that
    _cross_validation() is fed an array of parameters.

    Returns a float value with the smallest CV Error.
    """

    n = X.shape[0]

    # Split features into justifiable sizes

    n1 = np.rint(n * (1 - 1/np.log(n)))
    # n2 = n / np.log(n)

    temp = []

    for s in param:

        total = 0

        for i in range(len(nCV)):

            # Check if each loop we have a different permutation each time :(

            indices = np.random.permutation(X.shape[0])
            X1_idx, X2_idx = indices[:n1], indices[n1:]
            X1, X2 = X[X1_idx,:], X[X2_idx,:]
            total += np.linalg.norm(transform(X1, s) - np.cov(X2, rowvar=False, bias=True))

        total = total * (1/nCV)
        temp.append(total)
    
    return min(temp)
