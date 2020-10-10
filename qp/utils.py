"""Utility functions for the qp package"""

import numpy as np

from scipy import stats as sps
from scipy.interpolate import interp1d, splev, splrep
from scipy.integrate import quad
import sys

epsilon = sys.float_info.epsilon
infty = sys.float_info.max * epsilon
lims = (epsilon, 1.)

def sandwich(in_arr, ends):
    """
    Adds given values to the ends of a 1D array

    Parameters
    ----------
    in_arr: numpy.ndarray, float
        original array
    ends: numpy.ndarray or tuple or list, float or numpy.ndarray, float
        values to be added to the beginning and end

    Returns
    -------
    out_arr: numpy.ndarray, float
        array with front and back concatenations
    """
    if isinstance(ends[0], np.ndarray):
        prepend = len(ends[0])
    else:
        prepend = 1
    if isinstance(ends[-1], np.ndarray):
        append = -1 * len(ends[-1])
    else:
        append = -1
    out_arr = np.zeros(prepend + len(in_arr) - append)
    out_arr[:prepend] = ends[0]
    out_arr[prepend:append] = in_arr
    out_arr[append:] = ends[-1]
    return out_arr

def safelog(arr, threshold=epsilon):
    """
    Takes the natural logarithm of an array of potentially non-positive numbers

    Parameters
    ----------
    arr: numpy.ndarray, float
        values to be logged
    threshold: float
        small, positive value to replace zeros and negative numbers

    Returns
    -------
    logged: numpy.ndarray
        logarithms, with approximation in place of zeros and negative numbers
    """
    shape = np.shape(arr)
    flat = arr.flatten()
    logged = np.log(np.array([max(a, threshold) for a in flat])).reshape(shape)
    return logged

def normalize_integral(in_data, vb=False):
    """
    Normalizes integrals of PDF evaluations on a grid

    Parameters
    ----------
    in_data: None or tuple, numpy.ndarray, float
        tuple of points x at which function is evaluated and the PDF y at those
        points
    vb: boolean, optional
        be careful and print progress to stdout?

    Returns
    -------
    out_data: tuple, numpy.ndarray, float
        tuple of ordered input x and normalized y
    """
    if in_data is None:
        return in_data
    (x, y) = in_data
    # if vb:
    a = x.argsort()
    #     try:
    #         assert np.array_equal(x[a], x.sort())
    #     except AssertionError:
    x.sort()
    y = y[a]
    dx = x[1:] - x[:-1]
    my = (y[1:] + y[:-1]) / 2.
    norm = np.dot(my, dx)
    y = y / norm
    if vb:
        try:
            my = (y[1:] + y[:-1]) / 2.
            assert np.isclose(np.dot(my, dx), 1.)
        except AssertionError:
            print('`qp.utils.normalize_integral`: broken integral = '+str((my, dx)))
            assert False
    out_data = (x, y)
    return out_data

def evaluate_samples(in_data, bw_method=None, vb=False):
    """
    Produces PDF values given samples

    Parameters
    ----------
    in_data: numpy.ndarray, float
        samples x from the PDF
    bw_method: string or scalar or callable function, optional
        `scipy.stats.gaussian_kde` bandwidth methods: 'scott', 'silverman'
    vb: boolean, optional
        be careful and print progress to stdout?

    Returns
    -------
    out_data: tuple, float
        sorted samples x and corresponding PDF values y
    """
    x = in_data
    x.sort()
    kde = sps.gaussian_kde(x, bw_method)
    if vb:
        print('`qp.utils.evaluate_samples` made a KDE with bandwidth = '+str(kde.factor))
    y = kde(x)
    out_data = (x, y)
    return out_data

def evaluate_histogram(in_data, threshold=epsilon, vb=False):
    """
    Produces PDF values given samples

    Parameters
    ----------
    in_data: None or tuple, numpy.ndarray, float
        tuple of (n+1) bin endpoints x and (n) CDF y between endpoints
    threshold: float, optional

    vb: boolean, optional
        be careful and print progress to stdout?

    Returns
    -------
    out_data: tuple, float
        sorted samples x and corresponding PDF values y
    """
    (x, y) = in_data
    dx = threshold
    xs = np.zeros(2 * len(y))
    ys = xs
    xs[::2] = x[:-1] + dx
    xs[1::2] = x[1:] - dx
    ys = np.repeat(y, 2)
    xs = sandwich(xs, (x[0] - dx, x[-1] + dx))
    ys = sandwich(ys, (threshold, threshold))
    if vb:
        try:
            assert np.all(ys >= threshold)
        except AssertionError:
            print('broken self-evaluations in `qp.utils.evaluate_histogram`: '+str((xs, ys)))
            assert False
    out_data = (xs, ys)
    return out_data

def normalize_histogram(in_data, threshold=epsilon, vb=False):
    """
    Normalizes histogram parametrizations

    Parameters
    ----------
    in_data: None or tuple, numpy.ndarray, float
        tuple of (n+1) bin endpoints x and (n) CDF y between endpoints
    threshold: float, optional
        optional minimum threshold
    vb: boolean, optional
        be careful and print progress to stdout?

    Returns
    -------
    out_data: tuple, numpy.ndarray, float
        tuple of input x and normalized y
    """
    if in_data is None:
        return in_data
    (x, y) = in_data
    dx = x[1:] - x[:-1]
    y[y < threshold] = threshold
    y /= np.dot(y, dx)
    if vb:
        try:
            assert np.isclose(np.dot(y, dx), 1.)
        except AssertionError:
            print('`qp.utils.normalize_histogram`: broken integral = '+str(np.dot(y, dx)))
            assert False
    out_data = (x, y)
    return out_data

def normalize_gridded(in_data, thresholds=(epsilon, infty)):
    """
    Removes extreme values from gridded parametrizations

    Parameters
    ----------
    in_data: None or tuple, numpy.ndarray, float
        tuple of points x at which function is evaluated and the PDF y at those
        points
    thresholds: tuple, float, optional
        optional min/max thresholds for normalization

    Returns
    -------
    out_data: tuple, numpy.ndarray, float
        tuple of input x and normalized y
    """
    if in_data is None:
        return in_data
    (x, y) = in_data
    y[y < thresholds[0]] = thresholds[0]
    y[y > thresholds[-1]] = thresholds[-1]
    out_data = (x, y)
    return out_data

def evaluate_quantiles(in_data, threshold=epsilon, vb=False):
    """
    Estimates PDF values given quantile information

    Parameters
    ----------
    in_data: tuple, numpy.ndarray, float
        tuple of CDF values iy and values x at which those CDFs are achieved
    threshold: float, optional
        optional minimum threshold for CDF difference
    vb: boolean, optional
        be careful and print progress to stdout?

    Returns
    -------
    out_data: tuple, numpy.ndarray, float
        values xs and corresponding PDF values ys
    """
    (iy, x) = in_data
    dx = x[1:] - x[:-1]
    if vb:
        try:
            assert np.all(dx > threshold)
        except AssertionError:
            print('broken quantile locations in `qp.utils.evaluate_quantiles`: '+str(x))
            assert False
    diy = iy[1:] - iy[:-1]
    if vb:
        try:
            assert np.all(diy > threshold)
        except AssertionError:
            print('broken CDF values in `qp.utils.evaluate_quantiles`: '+str(iy))
            assert False
    y = diy / dx
    (xs, ys) = evaluate_histogram((x, y), threshold=threshold, vb=vb)
    if vb:
        print('input shape: '+str((len(x), len(y)))+', output shape: '+str((len(xs), len(ys))))
    #     try:
    #         assert (np.all(xs > threshold) and np.all(ys > threshold))
    #     except AssertionError:
    #         print('broken quantile self-evaluations in `qp.utils.evaluate_quantiles`: '+str((xs, ys)))
    #         assert False
    # ys = ys[1:-1]
    # xs = xs[1:-1]
    # ys = sandwich(ys, (threshold, threshold))
    # x_min = xs[0] - 2 * iy[0] / y[0]
    # x_max = xs[-1] + 2 * iy[-1] / y[-1]
    # xs = sandwich(xs, (x_min, x_max))
    out_data = (xs[1:-1], ys[1:-1])
    return out_data

def normalize_quantiles(in_data, threshold=epsilon, vb=False):
    """
    Evaluates PDF from quantiles including endpoints from linear extrapolation

    Parameters
    ----------
    in_data: tuple, numpy.ndarray, float
        tuple of CDF values iy corresponding to quantiles and the points x at
        which those CDF values are achieved
    threshold: float, optional
        optional minimum threshold for PDF
    vb: boolean, optional
        be careful and print progress to stdout?

    Returns
    -------
    out_data: tuple, ndarray, float
        tuple of values x at which CDF is achieved, including extrema, and
        normalized PDF values y at x
    """
    (iy, x) = in_data
    (xs, ys) = evaluate_quantiles((iy, x), vb=vb)
    # xs = xs[1:-1]
    # ys = ys[1:-1]
    x_min = xs[0] - 2 * iy[0] / ys[0]
    x_max = xs[-1] + 2 * (1. - iy[-1]) / ys[-1]
    xs = sandwich(xs, (x_min, x_max))
    ys = sandwich(ys, (threshold, threshold))
    out_data = (xs, ys)
    return out_data


def normalize_interp1d(xvals, yvals, limits, **kwargs):
    """
    Normalize a set of 1D interpolators

    Parameters
    ----------
    xvals : array-like
        X-values used for the interpolation
    yvals : array-like
        Y-values used for the interpolation
    limits : tuple (2)
        Lower and Upper limits of integration

    Keywords
    --------
    Passed to the `scipy.quad` intergation function

    Returns
    -------
    ynorm: array-like
        Normalized y-vals
    """

    def row_integral(irow):
        return quad(interp1d(xvals[irow], yvals[irow], **kwargs), limits[0], limits[1])[0]

    vv = np.vectorize(row_integral)
    integrals = vv(np.arange(xvals.shape[0]))
    return (yvals.T / integrals).T


def normalize_spline(xvals, yvals, limits, **kwargs):
    """
    Normalize a set of 1D interpolators

    Parameters
    ----------
    xvals : array-like
        X-values used for the spline
    yvals : array-like
        Y-values used for the spline
    limits : tuple (2)
        Lower and Upper limits of integration

    Keywords
    --------
    Passed to the `scipy.quad` intergation function

    Returns
    -------
    ynorm: array-like
        Normalized y-vals
    """

    def row_integral(irow):
        spline = lambda xv : splev(xv, splrep(xvals[irow], yvals[irow]))
        return quad(spline, limits[0], limits[1], **kwargs)[0]

    vv = np.vectorize(row_integral)
    integrals = vv(np.arange(xvals.shape[0]))
    return (yvals.T / integrals).T


def build_splines(xvals, yvals):
    """
    Build a set of 1D spline representations

    Parameters
    ----------
    xvals : array-like
        X-values used for the spline
    yvals : array-like
        Y-values used for the spline

    Returns
    -------
    splx : array-like
        Spline knot xvalues
    sply : array-like
        Spline knot yvalues
    spln : array-like
        Spline knot order paramters
    """
    l_x = []
    l_y = []
    l_n = []
    for xrow, yrow in zip(xvals, yvals):
        rep = splrep(xrow, yrow)
        l_x.append(rep[0])
        l_y.append(rep[1])
        l_n.append(rep[2])
    return np.vstack(l_x), np.vstack(l_y), np.vstack(l_n)


def build_kdes(samples, **kwargs):
    """
    Build a set of Gaussian Kernal Density Estimates

    Parameters
    ----------
    samples : array-like
        X-values used for the spline

    Keywords
    --------
    Passed to the `scipy.stats.gaussian_kde` constructor


    Retruns
    -------
    kdes : list of `scipy.stats.gaussian_kde` objects
    """
    return [ sps.gaussian_kde(row, **kwargs) for row in samples ]
