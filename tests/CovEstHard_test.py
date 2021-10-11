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
from CovarianceToolbox import CovEstHard

def test_cross_validation():
    """
    Cross validation test cases
    """
    features=[]
    param=[1,2,3]
    assert CovEstHard._cross_validation(features, param, 5) == None