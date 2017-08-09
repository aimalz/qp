"""
Notes
-----
TO DO: change dx --> dz (or delta)
"""

import numpy as np
import scipy as sp
from scipy import stats as sps
import sys
import bisect

import matplotlib.pyplot as plt

global epsilon
epsilon = sys.float_info.epsilon
global infty
infty = 100.

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
        result.append(cumsum / tot)
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

def safelog(arr, threshold=epsilon):
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
    logged = np.log(np.array([max(a, threshold) for a in flat])).reshape(shape)
    return logged

def normalize_integral(in_data, vb=False):
    """
    Normalizes integrals over full range from grid

    Parameters
    ----------
    in_data: None or tuple, ndarray, float
        tuple of points at which function is evaluated and the PDF at those points
    vb: boolean
        print progress to stdout?

    Returns
    -------
    (x, y): tuple, ndarray, float
        tuple of input x and normalized y
    """
    if in_data is None:
        return in_data
    (x, y) = in_data
    a = x.argsort()
    x.sort()
    ys = y[a]
    dx = x[1:] - x[:-1]
    dy = (ys[1:] + ys[:-1]) / 2.
    assert(len(dx) == len(dy))
    norm = np.dot(dy, dx)
    y = y / norm
    return(x, y)

def normalize_gridded(in_data, vb=True):
    """
    Normalizes gridded parametrizations assuming evenly spaced grid

    Parameters
    ----------
    in_data: None or tuple, ndarray, float
        tuple of points at which function is evaluated and the PDF at those points
    vb: boolean
        print progress to stdout?

    Returns
    -------
    (x, y): tuple, ndarray, float
        tuple of input x and normalized y
    """
    if in_data is None:
        return in_data
    (x, y) = in_data
    y[y < epsilon] = epsilon
    return (x, y)

def normalize_histogram(in_data, vb=True):
    """
    Normalizes histogram parametrizations

    Parameters
    ----------
    in_data: None or tuple, ndarray, float
        tuple of (n+1) bin endpoints and (n) CDF between endpoints
    vb: boolean
        print progress to stdout?

    Returns
    -------
    (x, y): tuple, ndarray, float
        tuple of input x and normalized y
    """
    if in_data is None:
        return in_data
    (x, y) = in_data
    delta = x[1:] - x[:-1]
    # if vb: print(np.sum(y * delta))
    y[y < epsilon] = epsilon
    y /= np.dot(y, delta)
    # if vb: print(np.sum(y * delta))
    return (x, y)

def evaluate_quantiles((qs, xs)):
    """
    Produces PDF values given quantile information

    Parameters
    ----------
    qs: ndarray, float
        CDF values
    xs: ndarray, float
        quantile values

    Returns
    -------
    (x, y): tuple, float
        quantile values and corresponding PDF

    Notes
    -----
    TO DO: make this use linear interpolation instead of piecewise constant
    """
    # q = np.append(q, np.array([1.]))
    # qs = np.append(np.array([0.]), q)
    # norm = max(qs) - min(qs)
    dq = qs[1:] - qs[:-1]
    # xs = np.append(x, np.array([infty]))
    # xs = np.append(np.array([-1. * infty]), x)
    dx = xs[1:] - xs[:-1]
    mx = (xs[1:] + xs[:-1]) / 2.
    y = dq / dx
    # print(np.dot(y, dx))
    # y *= norm
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

    Notes
    -----
    This shouldn't be necessary at all, see qp.PDF.interpolate notes
    """
    x = (xp[1:] + xp[:-1]) / 2.
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

def calculate_moment(p, N, using=None, limits=(-10.0,10.0), dx=0.01, vb=False):
    """
    Calculates moments of a distribution

    Parameters
    ----------
    p: qp.PDF object
        the PDF whose moment will be calculated
    N: int
        order of the moment to be calculated
    limits: tuple of floats
        endpoints of integration interval over which to calculate moments
    dx: float
        resolution of integration grid

    Returns
    -------
    M: float
        values of the moment
    """
    if using is None:
        using = p.first
    # Make a grid from the limits and resolution
    grid = np.linspace(limits[0], limits[1], int((limits[1]-limits[0])/dx))
    grid_to_N = grid ** N
    # Evaluate the functions on the grid
    pe = p.evaluate(grid, using=using, vb=vb)[1]
    # pe = normalize_gridded(pe)[1]
    # calculate the moment
    M = dx * np.dot(grid_to_N, pe)
    return M

def calculate_kl_divergence(p, q, limits=(-10.0,10.0), dx=0.01, vb=False):
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

    Notes
    -----
    TO DO: change this to calculate_kld
    """
    # Make a grid from the limits and resolution
    grid = np.linspace(limits[0], limits[1], int((limits[1]-limits[0])/dx))
    # Evaluate the functions on the grid
    pe = p.evaluate(grid, vb=vb)[1]
    qe = q.evaluate(grid, vb=vb)[1]
    # Normalize the evaluations, so that the integrals can be done
    # (very approximately!) by simple summation:
    pn = pe / np.sum(pe)
    #denominator = max(np.sum(qe), epsilon)
    qn = qe / np.sum(qe)#denominator
    # Compute the log of the normalized PDFs
    logp = safelog(pn)
    logq = safelog(qn)
    # Calculate the KLD from q to p
    Dpq = np.sum(pn * (logp - logq))
    return Dpq

def calculate_rmse(p, q, limits=(-10.,10.), dx=0.01, vb=False):
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
    vb: boolean
        report on progress to stdout?

    Returns
    -------
    rms: float
        the value of the RMS error between `q` and `p`
    """
    # Make a grid from the limits and resolution
    npoints = int((limits[1] - limits[0]) / dx)
    grid = np.linspace(limits[0], limits[1], npoints)
    # Evaluate the functions on the grid
    pe = p.evaluate(grid, vb=vb)[1]
    qe = q.evaluate(grid, vb=vb)[1]
    # Calculate the RMS between p and q
    rms = np.sqrt(np.sum((pe - qe) ** 2) / npoints)
    return rms
