import numpy as np
import sys

def safelog(arr, threshold=sys.float_info.epsilon):
    """
    Takes log of array with zeroes.

    Parameters
    ----------
    arr: ndarray
        Values to be logged
    threshold: float
        Small, positive value to replace zeros and negative numbers

    Returns
    -------
    logged: ndarray
        Logged values, with approximation in place of zeros and negative numbers
    """
    shape = np.shape(arr)
    flat = arr.flatten()
    logged = np.log(np.array([max(a,threshold) for a in flat])).reshape(shape)
    return logged

def calculate_kl_divergence(qf, pf, grid):
    """
    Calculates Kullback-Leibler Divergence

    Parameters
    ----------
    qn: PDF.truth.pdf function
        True probability distribution function to which distance will be calculated, not necessarily normalized.
    pn: PDF.approximate function
        Probability distribution function whose distance from the truth will be calculated, not necessarily normalized.
    grid: ndarray
        Points at which to evaluate the true and approximated probabilities

    Returns
    -------
    klpq: float
        Value of the Kullback-Leibler Divergence from distribution pn to distribution qn
    """
    pn = pf(grid)
    qn = qf(grid)
    p = pn/np.sum(pn)
    q = qn/np.sum(qn)
    logp = safelog(p)
    logq = safelog(q)
    klpq = np.sum(p*(logp-logq))
    return(klpq)