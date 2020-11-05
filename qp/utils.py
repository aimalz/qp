"""Utility functions for the qp package"""

import numpy as np

from scipy import stats as sps
from scipy.interpolate import splev, splrep, interp1d
from scipy.integrate import quad
import sys

epsilon = sys.float_info.epsilon
infty = sys.float_info.max * epsilon
lims = (epsilon, 1.)


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
    return np.log(np.array(arr).clip(threshold, np.inf))


"""
def normalize_quantiles(in_data, threshold=epsilon, vb=False):
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
"""

def normalize_interp1d(xvals, yvals):
    """
    Normalize a set of 1D interpolators

    Parameters
    ----------
    xvals : array-like
        X-values used for the interpolation
    yvals : array-like
        Y-values used for the interpolation

    Returns
    -------
    ynorm: array-like
        Normalized y-vals
    """
    #def row_integral(irow):
    #    return quad(interp1d(xvals[irow], yvals[irow], **kwargs), limits[0], limits[1])[0]

    #vv = np.vectorize(row_integral)
    #integrals = vv(np.arange(xvals.shape[0]))
    integrals = np.sum(xvals[:,1:]*yvals[:,1:] - xvals[:,:-1]*yvals[:,1:], axis=1)
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



def evaluate_kdes(xvals, kdes):
    """
    Build a evaluate a set of kdes

    Parameters
    ----------
    xvals : array_like
        X-values used for the spline
    kdes : list of `sps.gaussian_kde`
        The kernel density estimates

    Returns
    -------
    yvals : array_like
        The kdes evaluated at the xvamls
    """
    return np.vstack([kde(xvals) for kde in kdes])



def interpolate_unfactored_x_multi_y(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    xvals : array_like (N)
        X-values used for the interpolation
    yvals : array_like (N, M)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (M, n)
        The interpoalted values
    """
    # This is kinda stupid, computes a lot of extra values, but it is vectorized
    return interp1d(xvals, yvals[row], **kwargs)(x).diagonal()



def interpolate_unfactored_multi_x_y(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    xvals : array_like (N, M)
        X-values used for the interpolation
    yvals : array_like (N)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (M, n)
        The interpoalted values
    """
    def single_row(xv, rv):
        return interp1d(xvals[rv], yvals, **kwargs)(xv)
    vv = np.vectorize(single_row)
    return vv(x, row)



def interpolate_unfactored_multi_x_multi_y(x, row, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    xvals : array_like (N, M)
        X-values used for the interpolation
    yvals : array_like (N, M)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (M, n)
        The interpoalted values
    """
    def single_row(xv, rv):
        return interp1d(xvals[rv], yvals[rv], **kwargs)(xv)

    vv = np.vectorize(single_row)
    return vv(x, row)



def interpolate_multi_x_multi_y(x, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_line (n)
        X values to interpolate at:
    xvals : array_like (M, N)
        X-values used for the interpolation
    yvals : array_like (M, N)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (M, n)
        The interpoalted values
    """
    xy_vals = np.hstack([xvals, yvals])
    nx = xvals.shape[-1]
    def single_row(xy_vals_):
        return interp1d(xy_vals_[0:nx], xy_vals_[nx:], **kwargs)(x)
    vv = np.vectorize(single_row, signature="(%i)->(%i)" % (xvals.shape[-1]+yvals.shape[-1], x.size))
    return vv(xy_vals)


def interpolate_x_multi_y(x, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_line (n)
        X values to interpolate at:
    xvals : array_like (N)
        X-values used for the interpolation
    yvals : array_like (M, N)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (M, n)
        The interpoalted values
    """
    return interp1d(xvals, yvals, **kwargs)(x)



def interpolate_multi_x_y(x, xvals, yvals, **kwargs):
    """
    Interpolate a set of values

    Parameters
    ----------
    x : array_line (n)
        X values to interpolate at:
    xvals : array_like (M, N)
        X-values used for the interpolation
    yvals : array_like (N)
        Y-avlues used for the inteolation

    Returns
    -------
    vals : array_like (M, n)
        The interpoalted values
    """
    #def single_row(xrow):
    #    idx = np.searchsorted(xrow, x, side='left').clip(1, xrow.size-1)
    #    x0 = xrow[idx-1]
    #    x1 = xrow[idx]
    #    f = (x1 - x)/(x1 - x0)
    #    y0 = yvals[idx-1]
    #    y1 = yvals[idx]
    #    return f*y1 + (1 - f)*y0
    def single_row(xrow):
        return interp1d(xrow, yvals, **kwargs)(x)
    vv = np.vectorize(single_row, signature="(%i)->(%i)" % (yvals.size, x.size))
    return vv(xvals)
