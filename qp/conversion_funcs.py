"""This module implements functions to convert distributions between various representations
These functions should then be registered with the `qp.ConversionDict` using `qp_add_mapping`.
That will allow the automated conversion mechanisms to work.
"""

import numpy as np

from sklearn import mixture

def convert_using_vals_at_x(in_dist, **kwargs):
    """Convert using a set of x and y values.

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Keywords
    --------
    xvals : `np.array`
        Locations at which the pdf is evaluated

    Returns
    -------
    data : `dict`
        The extracted data
    """
    xvals = kwargs.pop('xvals', None)
    if xvals is None: # pragma: no cover
        raise ValueError("To convert to convert_using_xy_vals you must specify xvals")
    yvals = in_dist.pdf(xvals)
    return dict(xvals=xvals, yvals=yvals)


def convert_using_xy_vals(in_dist, **kwargs):
    """Convert using a set of x and y values.

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Keywords
    --------
    xvals : `np.array`
        Locations at which the pdf is evaluated

    Returns
    -------
    data : `dict`
        The extracted data
    """
    xvals = kwargs.pop('xvals', None)
    if xvals is None: # pragma: no cover
        raise ValueError("To convert using convert_using_xy_vals you must specify xvals")
    yvals = in_dist.pdf(xvals)
    expand_x = np.ones(yvals.shape) * np.squeeze(xvals)
    return dict(xvals=expand_x, yvals=yvals)


def convert_using_samples(in_dist, **kwargs):
    """Convert using a set of values sampled from the PDF

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Keywords
    --------
    size : `int`
        Number of samples to generate

    Returns
    -------
    data : `dict`
        The extracted data
    """
    samples = in_dist.rvs(size=kwargs.pop('size', 1000))
    xvals = kwargs.pop('xvals')
    return dict(samples=samples, xvals=xvals, yvals=None)


def convert_using_hist_values(in_dist, **kwargs):
    """Convert using a set of values sampled from the PDF

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Keywords
    --------
    bins : `np.array`
        Histogram bin edges

    Returns
    -------
    data : `dict`
        The extracted data
    """
    bins = kwargs.pop('bins', None)
    if bins is None: # pragma: no cover
        raise ValueError("To convert using convert_using_hist_samples you must specify bins")
    bins, pdfs = in_dist.histogramize(bins)
    return dict(bins=bins, pdfs=pdfs)


def convert_using_hist_samples(in_dist, **kwargs):
    """Convert using a set of values samples that are then histogramed

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Keywords
    --------
    bins : `np.array`
        Histogram bin edges
    size : `int`
        Number of samples to generate

    Returns
    -------
    data : `dict`
        The extracted data
    """
    bins = kwargs.pop('bins', None)
    size = kwargs.pop('size', 1000)
    if bins is None: # pragma: no cover
        raise ValueError("To convert using convert_using_hist_samples you must specify bins")
    samples = in_dist.rvs(size=size)

    def hist_helper(sample):
        return np.histogram(sample, bins=bins)[0]
    vv = np.vectorize(hist_helper, signature="(%i)->(%i)" % (samples.shape[0], bins.size-1))
    pdfs = vv(samples)
    return dict(bins=bins, pdfs=pdfs)


def convert_using_quantiles(in_dist, **kwargs):
    """Convert using a set of quantiles and the locations at which they are reached

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Keywords
    --------
    quantiles : `np.array`
        Quantile values to use

    Returns
    -------
    data : `dict`
        The extracted data
    """
    quants = kwargs.pop('quants', None)
    if quants is None: # pragma: no cover
        raise ValueError("To convert using convert_using_quantiles you must specify quants")
    locs = in_dist.ppf(quants)
    return dict(quants=quants, locs=locs)


def convert_using_fit(in_dist, **kwargs): # pragma: no cover
    """Convert to a functional distribution by fitting it to a set of x and y values

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Keywords
    --------
    xvals : `np.array`
        Locations at which the pdf is evaluated

    Returns
    -------
    data : `dict`
        The extracted data
    """
    raise NotImplementedError('convert_using_fit')
    #xvals = kwargs.pop('xvals', None)
    #if xvals is None:
    #   raise ValueError("To convert using convert_using_fit you must specify xvals")
    ##vals = in_dist.pdf(xvals)


def convert_using_mixmod_fit_samples(in_dist, **kwargs):
    """Convert to a mixture model using a set of values sample from the pdf

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Keywords
    --------
    ncomps : `int`
        Number of components in mixture model to use
    nsamples : `int`
        Number of samples to generate

    Returns
    -------
    data : `dict`
        The extracted data
    """
    n_comps = kwargs.pop('ncomps', 3)
    n_sample = kwargs.pop('nsamples', 1000)
    #samples = in_dist.rvs(size=(in_dist.npdf, n_sample))
    samples = in_dist.rvs(size=n_sample)
    def mixmod_helper(samps):
        estimator = mixture.GaussianMixture(n_components=n_comps)
        estimator.fit(samps.reshape(-1, 1))
        weights = estimator.weights_
        means = estimator.means_[:, 0]
        stdevs = np.sqrt(estimator.covariances_[:, 0, 0])
        ov = np.vstack([weights, means, stdevs])
        return ov

    vv = np.vectorize(mixmod_helper, signature="(%i)->(3,%i)" % (n_sample, n_comps))
    fit_vals = vv(samples)
    return dict(weights=fit_vals[:,0,:], means=fit_vals[:,1,:], stds=fit_vals[:,2,:])
