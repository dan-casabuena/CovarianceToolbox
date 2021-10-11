from operator import truediv
import numpy as np
import pandas as pd

def invisible_datamatrix(A, fname):
    """
    Function which indicates if a matrix is a cleaned, viable matrix
    """

    cond1 = (type(A) == np.ndarray or type(A) == pd.Dataframe)
    cond2 = not np.isinf(A)
    cond3 = not np.isnan(A)

    if cond1 and cond2 and cond3:
        return True
    raise TypeError # Input matrix is invalid.