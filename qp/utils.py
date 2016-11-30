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

def calculate_kl_divergence(pf, qf, limits=(-10.,10.), dx=0.01):
    """
    Calculates Kullback-Leibler Divergence

    Parameters
    ----------
    pf: PDF object
        Probability distribution function whose distance of its approximation _from_ the truth of qf will be calculated.
    qf: PDF object
        Probability distribution function whose distance of its truth _to_ the approximation of pf will be calculated.
    limits: tuple of floats
        Endpoints of integration interval in which to calculate KLD
    dx: float
        resolution of integration grid

    Returns
    -------
    klpq: float
        Value of the Kullback-Leibler Divergence from the approximation of qf to the truth of pf
    """
    #Make a grid from the limits and resolution
    grid = np.linspace(limits[0], limits[1], int((limits[1]-limits[0])/dx))
    #Evaluate the functions on the grid
    pe = pf.approximate(grid)
    qe = qf.evaluate(grid)
    #Normalize the evaluations
    pn = pe/np.sum(pe)
    qn = qe/np.sum(qe)
    #Store the log of the normalizations
    logp = safelog(pn)
    logq = safelog(qn)
    #Calculate the KLD from q to p
    klpq = np.sum(pn*(logp-logq))
    return(klpq)

def calculate_rms(pf, qf, limits=(-10.,10.), dx=0.01):
    """
    Calculates Root Mean Square Error

    Parameters
    ----------
    pf: PDF object
        Probability distribution function whose distance between its truth and the approximation of qf will be calculated.
    qf: PDF object
        Probability distribution function whose distance between its approximation and the truth of pf will be calculated.
    limits: tuple of floats
        Endpoints of integration interval in which to calculate RMS
    dx: float
        resolution of integration grid

    Returns
    -------
    rms: float
        Value of the root mean square error between the approximation of qf and the truth of pf
    """
    #Make a grid from the limits and resolution
    npoints = int((limits[1]-limits[0])/dx)
    grid = np.linspace(limits[0], limits[1], npoints)
    #Evaluate the functions on the grid
    pe = pf.evaluate(grid)
    qe = qf.approximate(grid)
    #Calculate the RMS between p and q
    rms = np.sqrt(np.sum((pe-qe)**2)/npoints)
    return(rms)