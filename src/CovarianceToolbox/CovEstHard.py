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

def estimate(features: Union[np.ndarray, pd.DataFrame, List[List]], threshold: Union[int, float] = None, nCV=10, parallel=False):
    """
    TODO Implement Docstring
    """

    try:
        if type(features) == list:
            features = np.array(features)
    except TypeError:
        pass
        
    if threshold is None:
        threshold = np.sqrt(np.log(features.shape[0]) / features.shape[1])
    else:
        threshold = threshold
        #Do the hustle and split the features into two
        #threshold = _cross_validation(features)

    #DEBUG if threshold actually changes value here... don't like the variable referencing here

    n = features.shape[0]
    p = features.shape[1]

    #Do we need these? I don't think so lmao

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

def _cross_validation(self):
    """
    TODO Implement Docstring
    """
    

class DimensionError(Exception):
    def __init__(self, message='Inputted data has dimensions not supported by package.'):
        self.message = message
    #Fix

        