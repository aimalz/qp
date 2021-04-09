"""Utility functions for the qp package"""

import numpy as np

from scipy import stats as sps
from scipy import linalg as sla
from scipy.interpolate import interp1d
from scipy.special import voigt_profile
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


_ = """
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

def edge_to_center(edges):
    """Return the centers of a set of bins given the edges"""
    return 0.5*(edges[1:] + edges[:-1])


def bin_widths(edges):
    """Return the widths of a set of bins given the edges"""
    return edges[1:] - edges[:-1]


def get_bin_indices(bins, x):
    """Return the bin indexes for a set of values

    If the bins are equal width this will use arithmatic,
    If the bins are not equal width this will use a binary search
    """
    widths = bin_widths(bins)
    if np.allclose(widths, widths[0]):
        idx = np.atleast_1d(np.floor((x-bins[0])/widths[0]).astype(int))
    else:
        idx = np.atleast_1d(np.searchsorted(bins, x, side='left')-1)
    mask = (idx >= 0) * (idx < bins.size-1)
    np.putmask(idx, 1-mask, 0)
    xshape = np.shape(x)
    return idx.reshape(xshape), mask.reshape(xshape)


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

    Returns
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


def evaluate_hist_x_multi_y(x, row, bins, vals):
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    bins : array_like (N)
        X-values used for the interpolation
    vals : array_like (N, M)
        Y-avlues used for the inteolation

    Returns
    -------
    out : array_like (M, n)
        The histogram values
    """
    idx, mask = get_bin_indices(bins, x)
    mask = np.ones((row.size, 1)) * mask
    return np.where(mask.flatten(), vals[:,idx][row].flatten(), 0)


def evaluate_unfactored_hist_x_multi_y(x, row, bins, vals):
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    bins : array_like (N)
        X-values used for the interpolation
    vals : array_like (N, M)
        Y-avlues used for the inteolation

    Returns
    -------
    out : array_like (M, n)
        The histogram values
    """
    idx, mask = get_bin_indices(bins, x)
    def evaluate_row(idxv, maskv, rv):
        return np.where(maskv, vals[rv, idxv], 0)
    vv = np.vectorize(evaluate_row)
    return vv(idx, mask, row)


def evaluate_hist_multi_x_multi_y(x, row, bins, vals):
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    bins : array_like (N, M)
        X-values used for the interpolation
    vals : array_like (N, M)
        Y-avlues used for the inteolation

    Returns
    -------
    out : array_like (M, n)
        The histogram values
    """
    n_vals = bins.shape[-1] - 1
    def evaluate_row(rv):
        idx, mask = get_bin_indices(bins[rv].flatten(), x)
        return np.where(mask, np.squeeze(vals[rv])[idx.clip(0, n_vals-1)], 0).flatten()
    vv = np.vectorize(evaluate_row, signature="(1)->(%i)" % (x.size))
    return vv(np.expand_dims(row, -1)).flatten()


def evaluate_unfactored_hist_multi_x_multi_y(x, row, bins, vals):
    """
    Evaluate a set of values from histograms

    Parameters
    ----------
    x : array_like (n)
        X values to interpolate at
    row : array_like (n)
        Which rows to interpolate at
    bins : array_like (N, M)
        X-values used for the interpolation
    vals : array_like (N, M)
        Y-avlues used for the inteolation

    Returns
    -------
    out : array_like (M, n)
        The histogram values
    """
    n_vals = bins.shape[-1] - 1
    def evaluate_row(xv, rv):
        idx, mask = get_bin_indices(bins[rv], xv)
        return np.where(mask, vals[rv,idx.clip(0, n_vals-1)], 0)
    vv = np.vectorize(evaluate_row)
    return vv(x, row)


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


def profile(x_data, y_data, x_bins, std=True):
    """Make a 'profile' plot

    Paramters
    ---------
    x_data : array_like (n)
        The x-values
    y_data : array_like (n)
        The y-values
    x_bins : array_like (nbins+1)
        The values of the bin edges
    std : bool
        If true, return the standard deviations, if false return the errors on the means

    Returns
    -------
    vals : array_like (nbins)
        The means
    errs : array_like (nbins)
        The standard deviations or errors on the means
    """
    idx, mask = get_bin_indices(x_bins, x_data)
    count = np.zeros(x_bins.size-1)
    vals = np.zeros(x_bins.size-1)
    errs = np.zeros(x_bins.size-1)
    for i in range(x_bins.size-1):
        mask_col = mask * (idx == i)
        count[i] = mask_col.sum()
        if mask_col.sum() == 0:  #pragma: no cover
            vals[i] = np.nan
            errs[i] = np.nan
            continue
        masked_vals = y_data[mask_col]
        vals[i] = masked_vals.mean()
        errs[i] = masked_vals.std()
    if not std:
        errs /= np.sqrt(count)
    return vals, errs

def reshape_to_pdf_size(vals, split_dim):
    """Reshape an array to match the number of PDFs in a distribution
    Parameters
    ----------
    vals : array
        The input array
    split_dim : int
        The dimension at which to split between pdf indices and per_pdf indices
    Returns
    -------
    out : array
        The reshaped array
    """
    # print(vals)
    # print(vals.shape)
    # print(split_dim)
    in_shape = vals.shape
    npdf = np.product(in_shape[:split_dim])
    per_pdf = in_shape[split_dim:]
    out_shape = np.hstack([npdf, per_pdf])
    return vals.reshape(out_shape)


def reshape_to_pdf_shape(vals, pdf_shape, per_pdf):
    """Reshape an array to match the shape of PDFs in a distribution

    Parameters
    ----------
    vals : array
        The input array
    pdf_shape : int
        The shape for the pdfs
    per_pdf : int or array_like
        The shape per pdf

    Returns
    -------
    out : array
        The reshaped array
    """
    outshape = np.hstack([pdf_shape, per_pdf])
    return vals.reshape(outshape)


def create_voigt_basis(zfine, mu, Nmu, sigma, Nsigma, Nv, cut=1.e-5):
    """
    Creates a gaussian-voigt dictionary at the same resolution as the original PDF

    :param float zfine: the x-axis for the PDF, the redshift resolution
    :param float mu: [min_mu, max_mu], range of mean for gaussian
    :param int Nmu: Number of values between min_mu and max_mu
    :param float sigma: [min_sigma, max_sigma], range of variance for gaussian
    :param int Nsigma: Number of values between min_sigma and max_sigma
    :param Nv: Number of Voigt profiles per gaussian at given position mu and sigma
    :param float cut: Lower cut for gaussians

    :return: Dictionary as numpy array with shape (len(zfine), Nmu*Nsigma*Nv)
    :rtype: float

    """

    zmid = np.linspace(mu[0], mu[1], Nmu)
    sig = np.linspace(sigma[0], sigma[1], Nsigma)
    gamma = np.linspace(0, 0.5, Nv)
    NA = Nmu * Nsigma * Nv
    Npdf = len(zfine)
    A = np.zeros((Npdf, NA))
    kk = 0
    for i in range(Nmu):
        for j in range(Nsigma):
            for k in range(Nv):
                #pdft = 1. * exp(-((zfine - zmid[i]) ** 2) / (2.*sig[j]*sig[j]))
                pdft = voigt_profile(zfine - zmid[i], sig[j], gamma[k])
                pdft = np.where(pdft >= cut, pdft, 0.)
                A[:, kk] = pdft / sla.norm(pdft)
                kk += 1
    return A

def sparse_basis(dictionary, query_vec, n_basis, tolerance=None):
    """
    Compute sparse representation of a vector given Dictionary  (basis)
    for a given tolerance or number of basis. It uses Cholesky decomposition to speed the process and to
    solve the linear operations adapted from Rubinstein, R., Zibulevsky, M. and Elad, M., Technical Report - CS
    Technion, April 2008

    :param float dictionary: Array with all basis on each column, must has shape (len(vector), total basis) and each column must have euclidean l-2 norm equal to 1
    :param float query_vec: vector of which a sparse representation is desired
    :param int n_basis: number of desired basis
    :param float tolerance: tolerance desired if n_basis is not needed to be fixed, must input a large number for n_basis to assure achieving tolerance

    :return: indices, values (2 arrays one with the position and the second with the coefficients)
    """

    a_n = np.zeros(dictionary.shape[1])
    machine_eps = np.finfo(dictionary.dtype).eps
    alpha = np.dot(dictionary.T, query_vec)
    res = query_vec
    idxs = np.arange(dictionary.shape[1])  # keeping track of swapping
    L = np.zeros((n_basis, n_basis), dtype=dictionary.dtype)
    L[0, 0] = 1.

    for n_active in range(n_basis):
        lam = np.argmax(abs(np.dot(dictionary.T, res)))
        if lam < n_active or alpha[lam] ** 2 < machine_eps:
            n_active -= 1
            break
        if n_active > 0:
            # Updates the Cholesky decomposition of dictionary
            L[n_active, :n_active] = np.dot(dictionary[:, :n_active].T, dictionary[:, lam])
            sla.solve_triangular(L[:n_active, :n_active], L[n_active, :n_active], lower=True, overwrite_b=True)
            v = sla.norm(L[n_active, :n_active]) ** 2
            if 1 - v <= machine_eps:
                print("Selected basis are dependent or normed are not unity")
                break
            L[n_active, n_active] = np.sqrt(1 - v)
        dictionary[:, [n_active, lam]] = dictionary[:, [lam, n_active]]
        alpha[[n_active, lam]] = alpha[[lam, n_active]]
        idxs[[n_active, lam]] = idxs[[lam, n_active]]
        # solves LL'x = query_vec as a composition of two triangular systems
        gamma = sla.cho_solve((L[:n_active + 1, :n_active + 1], True), alpha[:n_active + 1], overwrite_b=False)
        res = query_vec - np.dot(dictionary[:, :n_active + 1], gamma)
        if tolerance is not None and sla.norm(res) ** 2 <= tolerance:
            break
    a_n[idxs[:n_active + 1]] = gamma
    del dictionary
    #return a_n
    return idxs[:n_active + 1], gamma

def combine_int(Ncoef, Nbase):
    """
    combine index of base (up to 62500 bases) and value (16 bits integer with sign) in a 32 bit integer
    First half of word is for the value and second half for the index

    :param int Ncoef: Integer with sign to represent the value associated with a base, this is a sign 16 bits integer
    :param int Nbase: Integer representing the base, unsigned 16 bits integer
    :return: 32 bits integer
    """
    return (Ncoef << 16) | Nbase


def get_N(longN):
    """
    Extract coefficients fro the 32bits integer,
    Extract Ncoef and Nbase from 32 bit integer
    return (longN >> 16), longN & 0xffff

    :param int longN: input 32 bits integer

    :return: Ncoef, Nbase both 16 bits integer
    """
    return (longN >> 16), (longN & (2 ** 16 - 1))

def indices2shapes(sparse_indices, meta):
    """compute the Voigt shape parameters from the sparse index
    
    Parameters
    ----------
    sparse_index: `np.array`
        1D Array of indices for each object in the ensemble
    
    meta: `dict`
        Dictionary of metadata to decode the sparse indices
    """
    Nmu = meta['dims'][0]
    Nsigma = meta['dims'][1]
    Nv = meta['dims'][2]
    Ncoef = meta['dims'][3]
    zfine = meta['z']
    mu = meta['mu']
    sigma = meta['sig']

    means_array = np.linspace(mu[0], mu[1], Nmu)
    sig_array = np.linspace(sigma[0], sigma[1], Nsigma)
    gam_array = np.linspace(0, 0.5, Nv)

    #split the sparse indices into pairs (weight, basis_index)
    #for each sparse index corresponding to one of the basis function
    sp_ind = np.array(list(map(get_N, sparse_indices)))
    
    spi = sp_ind[:,0,:]
    dVals = 1./(Ncoef-1)
    vals = spi * dVals
    vals[:,0]=1.
    
    Dind2 = sp_ind[:,1,:]
    means = means_array[np.array(Dind2 / (Nsigma * Nv), int)]
    sigmas = sig_array[np.array((Dind2 % (Nsigma * Nv)) / Nv, int)]
    gammas =gam_array[np.array((Dind2 % (Nsigma * Nv)) % Nv, int)]
    
    return vals, means, sigmas, gammas


def build_sparse_representation(z, P, mu=None, Nmu=None, sig=None, Nsig=None, Nv=3, Nsparse=20, tol=1.e-10):
    #Note : the range for gamma is fixed to [0, 0.5] in create_voigt_basis
    Ntot = len(P)
    print("Total Galaxies = ", Ntot)
    dz = z[1] - z[0]
    print('dz = ', dz)
    if mu is None:
        mu = [min(z), max(z)]
    if Nmu is None:
        Nmu = len(z)
    if sig is None:
        max_sig = (max(z) - min(z)) / 12.
        min_sig = dz / 6.
        sig = [min_sig, max_sig]
    if Nsig is None:
        Nsig = int(np.ceil(2. * (max_sig - min_sig) / dz))

    print('Nmu, Nsig, Nv = ', '[', Nmu, ',', Nsig, ',', Nv, ']')
    print('Total bases in dictionary', Nmu * Nsig * Nv)

    #Create dictionary
    print('Creating Dictionary...')
    A = create_voigt_basis(z, mu, Nmu, sig, Nsig, Nv)
    bigD = {}

    Nsparse = 20
    Ncoef = 32001
    AA = np.linspace(0, 1, Ncoef)
    Da = AA[1] - AA[0]

    print('Nsparse (number of bases) = ', Nsparse)

    bigD['z'] = z
    bigD['mu'] = mu
    bigD['sig'] = sig
    bigD['dims'] = [Nmu, Nsig, Nv, Ncoef]
    bigD['N_SPARSE'] = Nsparse
    bigD['Ntot'] = Ntot

    print('Creating Sparse representation...')

    for k in range(Ntot):
        bigD[k] = {}
        try:
            pdf0 = P[k]
        except:
            continue
        
        Dind, Dval = sparse_basis(A, pdf0, Nsparse, tolerance=tol)

        if len(Dind) <= 1: continue
        bigD[k]['sparse'] = [Dind, Dval]
        if max(Dval) > 0:
            dval0=Dval[0]
            Dvalm = Dval / np.max(Dval)
            index = np.array(list(map(round, (Dvalm / Da))), dtype='int')
            index0=int(round(dval0/Da))
            index[0]=index0
        else:
            index = np.zeros(len(Dind), dtype='int')

        bigD[k]['sparse_ind'] = np.array(list(map(combine_int, index, Dind)))
    
        #swap back columns
        A[:, [Dind]] = A[:, [np.arange(len(Dind))]]

    #For now, extend the representation into a full 2D array
    #This may be removed eventually
    ALL = np.zeros((Ntot, Nsparse), dtype='int')
    for i in range(Ntot):
        if i in bigD:
            idd = bigD[i]['sparse_ind']
            ALL[i, 0:len(idd)] = idd
    print('done')
    return ALL, bigD
