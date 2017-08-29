"""
Notes
-----
TO DO: change dx --> dz (or delta)
"""

import numpy as np
import scipy as sp
from scipy import stats as sps
import sys
# import bisect

global epsilon
epsilon = sys.float_info.epsilon
global infty
infty = sys.float_info.max * epsilon
global lims
lims = (epsilon, 1.)

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
    # a = x.argsort()
    # x.sort()
    # ys = y[a]
    dx = x[1:] - x[:-1]
    dy = (y[1:] + y[:-1]) / 2.
    norm = np.dot(dy, dx)
    y = y / norm
    if vb:
        print('almost normalized integrals')
        dy = (y[1:] + y[:-1]) / 2.
        if not np.isclose(np.dot(dy, dx), 1.):
            print('broken integral = '+str(np.dot(dy, dx)))
            assert False
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
    y[y > infty] = infty
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

def normalize_quantiles((q, z), (x, y), vb=True):
    """
    Adds valid endpoints to quantile parametrization

    Parameters
    ----------
    q: numpy.ndarray, float
        CDF values corresponding to quantiles
    z: numpy.ndarray, float
        original quantiles
    x: numpy.ndarray, float
        averaged quantile values
    y: numpy.ndarray, float
        probability evaluated at averaged quantiles
    vb: boolean
        print progress to stdout?

    Returns
    -------
    (x, y): tuple, ndarray, float
        tuple of input x and normalized y

    Notes
    -----
    Finds actual endpoints via linear interpolation from evaluation
    """
    # nq = np.insert(q, [0, -1], (0., 1.))
    q = np.insert(q, 0, 0.)
    q = np.append(q, 1.)
    # nq = (q[1:] + q[:-1]) / 2.
    dq = q[1:] - q[:-1]
    # if vb:
    #     if not np.all(nq>0.)...
    xmin = z[0] - 2 * (dq[0] + dq[1] / 2. - y[0] * (x[0] - z[0])) / y[0]
    xmax = z[-1] + 2 * (dq[-1] + dq[-2]/2. - y[-1] * (z[-1] - x[-1])) / y[-1]
    x = np.insert(x, 0, xmin)
    x = np.append(x, xmax)
    y = np.insert(y, 0, 0.)
    y = np.append(y, 0.)
    return(x, y)

def evaluate_quantiles((qs, xs), vb=True):
    """
    Produces PDF values given quantile information

    Parameters
    ----------
    qs: ndarray, float
        CDF values
    xs: ndarray, float
        quantile values
    vb: Boolean
        print progress to stdout?

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
    if vb:
        if not np.all(dx>0.):
            print('broken delta quantile values: '+str(xs))
            assert(np.all(dx>0.))
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

def calculate_moment(p, N, using=None, limits=None, dx=0.01, vb=False):
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
    if limits is None:
        limits = p.limits
    if using is None:
        using = p.first
    # Make a grid from the limits and resolution
    grid = np.arange(limits[0], limits[1], dx)
    grid_to_N = grid ** N
    # Evaluate the functions on the grid
    pe = p.evaluate(grid, using=using, vb=vb)[1]
    # pe = normalize_gridded(pe)[1]
    # calculate the moment
    M = dx * np.dot(grid_to_N, pe)
    return M

def calculate_kl_divergence(p, q, limits=lims, dx=0.01, vb=False):
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
    TO DO: have this take number of points not dx!
    """
    # Make a grid from the limits and resolution
    grid = np.arange(limits[0], limits[1], dx)
    # Evaluate the functions on the grid and normalize
    pe = p.evaluate(grid, vb=vb, norm=True)
    pn = pe[1]
    qe = q.evaluate(grid, vb=vb, norm=True)
    qn = qe[1]
    # Normalize the evaluations, so that the integrals can be done
    # (very approximately!) by simple summation:
    # pn = pe / np.sum(pe)
    #denominator = max(np.sum(qe), epsilon)
    # qn = qe / np.sum(qe)#denominator
    # Compute the log of the normalized PDFs
    logquotient = safelog(pn / qn)
    # logp = safelog(pn)
    # logq = safelog(qn)
    # Calculate the KLD from q to p
    Dpq = np.dot(pn * logquotient, np.ones(len(grid)) * dx)
    if Dpq < 0.:
        print('broken KLD: '+str((Dpq, pn, qn, dx)))
        Dpq = epsilon
    return Dpq

def calculate_rmse(p, q, limits=lims, dx=0.01, vb=False):
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
