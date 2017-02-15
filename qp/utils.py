import numpy as np
import scipy as sp
from scipy import stats as sps
import sys
import bisect

import matplotlib.pyplot as plt

def cdf(weights):
    """
    Creates a normalized CDF from an arbitrary discrete distribution

    Parameters
    ----------
    weights: ndarray, float
        array of relative probabilities for classes

    Returns
    -------
    result: ndarray, float
        discrete CDF
    """
    tot = sum(weights)
    result = []
    cumsum = 0.
    for w in weights:
        cumsum += w
        result.append(cumsum/tot)
    return np.array(result)

def choice(pop, weights):
    """
    Samples classes from a discrete CDF

    Parameters
    ----------
    pop: ndarray or list, float or int or str
        possible classes to assign to sample
    weights: ndarray, float
        array of relative probabilities for classes

    Returns
    -------
    output: float or int or str
        the label on the class for the sample
    """
    assert len(pop) == len(weights)
    cdf_vals = cdf(weights)
    x = np.random.random()
    index = bisect.bisect(cdf_vals,x)
    output = pop[index]
    return output

def safelog(arr, threshold=sys.float_info.epsilon):
    """
    Takes the natural logarithm of an array that might contain zeroes.

    Parameters
    ----------
    arr: ndarray
        values to be logged
    threshold: float
        small, positive value to replace zeros and negative numbers

    Returns
    -------
    logged: ndarray
        logarithms, with approximation in place of zeros and negative numbers
    """
    shape = np.shape(arr)
    flat = arr.flatten()
    logged = np.log(np.array([max(a,threshold) for a in flat])).reshape(shape)
    return logged

def evaluate_quantiles((q, x), infty=100.):
    """
    Produces PDF values given quantile information

    Parameters
    ----------
    q: ndarray, float
        CDF values
    x: ndarray, float
        quantile values
    infty: float, optional
        value at which CDF(infty) = 0, CDF(-infty) = 0

    Returns
    -------
    (x, y): tuple, float
        quantile values and corresponding PDF
    """
    # q = np.append(q, np.array([1.]))
    # qs = np.append(np.array([0.]), q)
    qs = q
    dq = qs[1:]-qs[:-1]
    # xs = np.append(x, np.array([infty]))
    # xs = np.append(np.array([-1. * infty]), x)
    xs = x
    dx = xs[1:]-xs[:-1]
    mx = (xs[1:]+xs[:-1])/2.
    y = dq / dx
    return ((mx, y))

def evaluate_histogram((xp, y)):
    """
    Produces PDF values given histogram information

    Parameters
    ----------
    xp: ndarray, float
        bin endpoints
    y: ndarray, float
        CDFs over bins

    Returns
    -------
    (x, y): tuple, float
        bin midpoints and CDFs over bins
    """
    x = (xp[1:]+xp[:-1])/2.
    return((x, y))

def evaluate_samples(x):
    """
    Produces PDF values given samples

    Parameters
    ----------
    x: ndarray, float
        samples from the PDF

    Returns
    -------
    (sx, y): tuple, float
        sorted sample values and corresponding PDF values
    """
    sx = np.sort(x)
    # bandwidth = np.mean(sx[1:]-sx[:-1])
    kde = sps.gaussian_kde(x)# , bw_method=bandwidth)
    y = kde(sx)
    return ((sx, y))

def calculate_kl_divergence(p, q, limits=(-10.0,10.0), dx=0.01, vb=True):
    """
    Calculates the Kullback-Leibler Divergence between two PDFs.

    Parameters
    ----------
    p: PDF object
        probability distribution whose distance _from_ `q` will be calculated.
    q: PDF object
        probability distribution whose distance _to_ `p` will be calculated.
    limits: tuple of floats
        endpoints of integration interval in which to calculate KLD
    dx: float
        resolution of integration grid
    vb: boolean
        report on progress to stdout?

    Returns
    -------
    Dpq: float
        the value of the Kullback-Leibler Divergence from `q` to `p`
    """
    # Make a grid from the limits and resolution
    grid = np.linspace(limits[0], limits[1], int((limits[1]-limits[0])/dx))
    # Evaluate the functions on the grid
    pe = p.evaluate(grid, vb=vb)
    qe = q.evaluate(grid, vb=vb)
    # Normalize the evaluations, so that the integrals can be done
    # (very approximately!) by simple summation:
    pn = pe/np.sum(pe)
    qn = qe/np.sum(qe)
    # Compute the log of the normalized PDFs
    logp = safelog(pn)
    logq = safelog(qn)
    # Calculate the KLD from q to p
    Dpq = np.sum(pn*(logp-logq))
    return Dpq

def calculate_rms(p, q, limits=(-10.,10.), dx=0.01):
    """
    Calculates the Root Mean Square Error between two PDFs.

    Parameters
    ----------
    p: PDF object
        probability distribution function whose distance between its truth and the approximation of `q` will be calculated.
    q: PDF object
        probability distribution function whose distance between its approximation and the truth of `p` will be calculated.
    limits: tuple of floats
        endpoints of integration interval in which to calculate RMS
    dx: float
        resolution of integration grid

    Returns
    -------
    rms: float
        the value of the RMS error between `q` and `p`
    """
    # Make a grid from the limits and resolution
    npoints = int((limits[1]-limits[0])/dx)
    grid = np.linspace(limits[0], limits[1], npoints)
    # Evaluate the functions on the grid
    pe = p.evaluate(grid)
    qe = q.evaluate(grid)
    # Calculate the RMS between p and q
    rms = np.sqrt(np.sum((pe-qe)**2)/npoints)
    return rms
