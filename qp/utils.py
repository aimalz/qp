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

def calculate_kl_divergence(pf, qf, limits=(-10.0,10.0), dx=0.01, vb=True):
    """
    Calculates Kullback-Leibler Divergence between two PDFs.

    Parameters
    ----------
    pf: PDF object
        Probability distribution function whose distance _from_ qf will be calculated.
    qf: PDF object
        Probability distribution function whose distance _to_ pf will be calculated.
    limits: tuple of floats
        Endpoints of integration interval in which to calculate KLD
    dx: float
        Resolution of integration grid
    vb: boolean
        Report on progress to stdout?

    Returns
    -------
    klpq: float
        Value of the Kullback-Leibler Divergence from qf to pf
    """
    # Make a grid from the limits and resolution
    grid = np.linspace(limits[0], limits[1], int((limits[1]-limits[0])/dx))
    # Evaluate the functions on the grid
    pe = pf.evaluate(grid, vb=vb)
    qe = qf.evaluate(grid, vb=vb)
    # Normalize the evaluations, so that the integrals can be done
    # (very approximately!) by simple summation:
    pn = pe/np.sum(pe)
    qn = qe/np.sum(qe)
    # Compute the log of the normalized PDFs
    logp = safelog(pn)
    logq = safelog(qn)
    # Calculate the KLD from q to p
    klpq = np.sum(pn*(logp-logq))
    return(klpq)

def calculate_rms(pf, qf, limits=(-10.,10.), dx=0.01):
    """
    Calculates Root Mean Square Error between two PDFs.

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
    qe = qf.evaluate(grid)
    #Calculate the RMS between p and q
    rms = np.sqrt(np.sum((pe-qe)**2)/npoints)
    return(rms)
