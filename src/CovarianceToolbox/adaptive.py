import numpy as np
import pandas as pd
#from CovarianceToolbox import aux_invisible

def adaptive(X, thr=0.5, nCV=10, parallel=False):
    """
    Cai and Liu's variation of Bickel and Levina's hard thresholding.
    """
    #Preprocessing

    fname = 'adaptive'

    #checker1 = aux_invisible.invisible_datamatrix(X, fname)

    