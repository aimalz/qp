"""This module implements functions to convert distributions between various representations
These functions should then be registered with the `qp.ConversionDict` using `qp_add_mapping`.
That will allow the automated conversion mechanisms to work.
"""

import numpy as np
from sklearn import mixture

from .ensemble import Ensemble


def convert_using_xy_vals(in_dist, class_to, **kwargs):
    """Convert using a set of x and y values.
    Keywords
    --------
    xvals : `np.array`
        Locations at which the pdf is evaluated
    Remaining keywords are passed to class constructor.
    Returns
    -------
    dist : An distrubtion object of type class_to, instantiated using the x and y values
    """

    xvals = kwargs.pop('xvals', None)
    if xvals is None:
        raise ValueError("To convert to class %s using convert_using_xy_vals you must specify xvals" % class_to)
    yvals = in_dist.pdf(xvals)
    expand_x = np.ones(yvals.shape) * np.squeeze(xvals)
    return Ensemble(class_to, data=dict(xvals=expand_x, yvals=yvals), **kwargs)


def convert_using_samples(in_dist, class_to, **kwargs):
    """Convert using a set of values samples from the PDF
    Keywords
    --------
    size : `int`
        Number of samples to generate
    Remaining keywords are passed to class constructor.
    Returns
    -------
    dist : An distrubtion object of type class_to, instantiated using the x and y values
    """
    samples = in_dist.rvs(kwargs.pop('size', 10000))
    return Ensemble(class_to, data=dict(samples=samples), **kwargs)


def convert_using_hist_values(in_dist, class_to, **kwargs):
    """Convert using a set the CDF to make a histogram
    Keywords
    --------
    bins : `np.array`
        Histogram bin edges
    size : `int`
        Number of samples to generate
    Remaining keywords are passed to class constructor.
    Returns
    -------
    dist : An distrubtion object of type class_to, instantiated using the histogrammed samples
    """

    bins = kwargs.pop('bins', None)
    if bins is None:
        raise ValueError("To convert to class %s using convert_using_hist_samples you must specify bins" % class_to)
    hist = in_dist.histogramize(bins=bins)
    return Ensemble(class_to, data=dict(bins=hist[0], pdfs=hist[1]), **kwargs)


def convert_using_hist_samples(in_dist, class_to, **kwargs):
    """Convert using a set of values samples that are then histogramed
    Keywords
    --------
    bins : `np.array`
        Histogram bin edges
    size : `int`
        Number of samples to generate
    Remaining keywords are passed to class constructor.
    Returns
    -------
    dist : An distrubtion object of type class_to, instantiated using the histogrammed samples
    """

    bins = kwargs.pop('bins', None)
    if bins is None:
        raise ValueError("To convert to class %s using convert_using_hist_samples you must specify xvals" % class_to)
    hist = np.histogram(in_dist.rvs(kwargs.pop('size', 10000)), bins=bins)
    return Ensemble(class_to, data=dict(bins=hist[0], pdfs=hist[1]), **kwargs)


def convert_using_quantiles(in_dist, class_to, **kwargs):
    """Convert using a set of quantiles and the locations at which they are reached
    Keywords
    --------
    quantiles : `np.array`
        Quantile values to use
    Remaining keywords are passed to class constructor.
    Returns
    -------
    dist : An distrubtion object of type class_to, instantiated using the qunatile values and locations
    """

    quants = kwargs.pop('quants', None)
    if quants is None:
        raise ValueError("To convert to class %s using convert_using_quantiles you must specify quants" % class_to)
    locs = in_dist.ppf(quants)
    return Ensemble(class_to, data=dict(quants=quants, locs=locs), **kwargs)


def convert_using_fit(in_dist, class_to, **kwargs):
    """Convert to a functional distribution by fitting it to a set of x and y values
    Keywords
    --------
    xvals : `np.array`
        Locations at which the pdf is evaluated
    Remaining keywords are passed to class constructor.
    Returns
    -------
    dist : An distrubtion object of type class_to, instantiated by fitting to the samples.
    """
    raise NotImplementedError('convert_using_fit')
    #xvals = kwargs.pop('xvals', None)
    #if xvals is None:
    #   raise ValueError("To convert to class %s using convert_using_fit you must specify xvals" % class_to)
    ##vals = in_dist.pdf(xvals)


def convert_using_mixmod_fit_samples(in_dist, class_to, **kwargs):
    """Convert to a mixture model using a set of values sample from the pdf
    Keywords
    --------
    ncomps : `int`
        Number of components in mixture model to use
    nsamples : `int`
        Number of samples to generate
    Remaining keywords are passed to class constructor.
    Returns
    -------
    dist : An distrubtion object of type class_to, instantiated by fitting to the samples.
    """
    n_comps = kwargs.pop('ncomps', 5)
    n_sample = kwargs.pop('nsamples', 1000)
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
    return Ensemble(class_to, data=dict(weights=fit_vals[:,0,:], means=fit_vals[:,1,:], stds=fit_vals[:,2,:]))
