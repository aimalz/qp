"""This module implements functions to convert distributions between various representations
These functions should then be registered with the `qp.ConversionDict` using `qp_add_mapping`.
That will allow the automated conversion mechanisms to work.
"""
import numpy
import numpy as np

from sklearn import mixture
from .utils import create_voigt_basis, sparse_basis, combine_int, indices2shapes

def extract_vals_at_x(in_dist, **kwargs):
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
        raise ValueError("To convert to extract_xy_vals you must specify xvals")
    yvals = in_dist.pdf(xvals)
    return dict(xvals=xvals, yvals=yvals)


def extract_xy_vals(in_dist, **kwargs):
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
        raise ValueError("To convert using extract_xy_vals you must specify xvals")
    yvals = in_dist.pdf(xvals)
    expand_x = np.ones(yvals.shape) * np.squeeze(xvals)
    return dict(xvals=expand_x, yvals=yvals)


def extract_samples(in_dist, **kwargs):
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


def extract_hist_values(in_dist, **kwargs):
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
        raise ValueError("To convert using extract_hist_samples you must specify bins")
    bins, pdfs = in_dist.histogramize(bins)
    return dict(bins=bins, pdfs=pdfs)


def extract_hist_samples(in_dist, **kwargs):
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
        raise ValueError("To convert using extract_hist_samples you must specify bins")
    samples = in_dist.rvs(size=size)

    def hist_helper(sample):
        return np.histogram(sample, bins=bins)[0]
    vv = np.vectorize(hist_helper, signature="(%i)->(%i)" % (samples.shape[0], bins.size-1))
    pdfs = vv(samples)
    return dict(bins=bins, pdfs=pdfs)


def extract_quantiles(in_dist, **kwargs):
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
        raise ValueError("To convert using extract_quantiles you must specify quants")
    locs = in_dist.ppf(quants)
    return dict(quants=quants, locs=locs)


def extract_fit(in_dist, **kwargs): # pragma: no cover
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
    raise NotImplementedError('extract_fit')
    #xvals = kwargs.pop('xvals', None)
    #if xvals is None:
    #   raise ValueError("To convert using extract_fit you must specify xvals")
    ##vals = in_dist.pdf(xvals)


def extract_mixmod_fit_samples(in_dist, **kwargs):
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

def extract_voigt_mixmod(in_dist, **kwargs):
    """Convert to a voigt mixture model starting with a gaussian mixture model, 
    trivially by setting gammas to 0

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Returns
    -------
    data : `dict`
        The extracted data
    """
    objdata = in_dist.objdata()
    means = objdata['means']
    stds = objdata['stds']
    weights = objdata['weights']
    gammas = np.zeros_like(means)
    return dict(means=means, stds=stds, weights=weights, gammas=gammas)


def extract_voigt_xy(in_dist, **kwargs):
    """Build a voigt function basis and run a match-pursuit algorithm to fit gridded data

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Returns
    -------
    data : `dict`
        The extracted data as sparse indices, basis, and metadata to rebuild the basis
    """

    sparse_results = extract_voigt_xy_sparse(in_dist, **kwargs)
    indices = sparse_results['indices']
    meta = sparse_results['metadata']
    basis = sparse_results['basis']

    weights = []
    means = []
    stds = []
    gammas = []
    for ind in indices:
        w, m, s, g = indices2shapes(ind, meta)
        means.append(m)
        weights.append(w)
        stds.append(s)
        gammas.append(g)
    print(weights)
    return dict(means=np.asarray(means), stds=np.asarray(stds), weights=np.asarray(weights), gammas=np.asarray(gammas))
    
def extract_voigt_xy_sparse(in_dist, **kwargs):
    """Build a voigt function basis and run a match-pursuit algorithm to fit gridded data

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input distributions

    Returns
    -------
    data : `dict`
        The extracted data as shaped parameters means, stds, weights, gammas
    """

    yvals = in_dist.objdata()['yvals']
    
    default = in_dist.metadata()['xvals'][0]
    z = kwargs.pop('xvals', default)

    mu = [np.min(z), np.max(z)]
    mu = kwargs.pop('mu', mu)
    Nmu = kwargs.pop('Nmu', 250)
    Nv = kwargs.pop('Nv', 3)
    
    dz = np.min(np.diff(z))
    max_sig = (max(z) - min(z)) / 12.
    min_sig = dz / 6.
    Nsig = int(numpy.ceil(2. * (max_sig - min_sig) / dz))
    sig = [min_sig, max_sig]
    sig = kwargs.pop('sig', sig)
    Nsig = kwargs.pop('Nsig',Nsig)

    A = create_voigt_basis(z, mu, Nmu, sig, Nsig, Nv)

    toler = 1.e-10
    Nsparse = 20
    Ncoef = 32001
    AA = np.linspace(0, 1, Ncoef)
    Da = AA[1] - AA[0]

    sparse_ind = []
    bigD = {}
    bigD['z'] = z
    bigD['mu'] = mu
    bigD['sig'] = sig
    bigD['N_SPARSE'] = Nsparse
    bigD['Ncoef'] = Ncoef
    bigD['Nmu'] = Nmu
    bigD['Nsig'] = Nsig
    bigD['Nv'] = Nv
    #bigD['Ntot'] = Ntot

    for k, pdf0 in enumerate(yvals):
        #sparse_ind[k] = {}
        if sum(pdf0) > 0:
            pdf0 /= sum(pdf0)
        else:
            continue
        Dind, Dval = sparse_basis(A, pdf0, Nsparse)
        if len(Dind) <= 1: continue
        #sparse_ind[k]['sparse'] = [Dind, Dval]
        if max(Dval) > 0:
            dval0=Dval[0]
            Dvalm = Dval / max(Dval)
            index = np.array(list(map(round, (Dvalm / Da))), dtype='int')
            index0=int(round(dval0/Da))
            index[0]=index0
        else:
            index = zeros(len(Dind), dtype='int')
        #sparse_ind[k]['sparse_ind'] = np.array(map(combine_int, index, Dind))
        sparse_ind.append(np.array(list(map(combine_int, index, Dind))))

        #swap back columns
        A[:, [Dind]] = A[:, [np.arange(len(Dind))]]

    #print(sparse_ind)
    return dict(indices=sparse_ind, metadata=bigD, basis=A)
