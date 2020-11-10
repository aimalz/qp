"""This module implements continous distributions generators that inherit from the
`scipy.stats.rv_continuous` class

If you would like to add a sub-class, please read the instructions on subclassing
here:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html


Open questions:
1) At this time the normalization is not enforced for many of the PDF types.  It is assumed that
the user values give correct normalization.  We should think about this more.

2) At this time for most of the distributions, only the _pdf function is overridden.  This is all that
is required to inherit from `scipy.stats.rv_continuous`; however, providing implementations of some of
_logpdf, _cdf, _logcdf, _ppf, _rvs, _isf, _sf, _logsf could speed the code up a lot is some cases.

"""

import numpy as np
from numpy import asarray

from scipy.stats import rv_continuous
from scipy.stats._distn_infrastructure import rv_frozen, _moment_from_stats
from scipy.special import comb

from .plotting import plot_dist_pdf
from .conversion import qp_convert




class Pdf_gen:
    """Interface class to extend `scipy.stats.rv_continuous` with
    information needed for `qp`

    Notes
    -----
    Metadata are elements that are the same for all the PDFs
    These include the name and version of the PDF generation class, and
    possible data such as the bin edges used for histogram representations

    Object data are elements that differ for each PDFs
    """

    def __init__(self, *args, **kwargs):
        """C'tor"""
        # pylint: disable=unused-argument
        self._metadata = {}
        self._objdata = {}
        self._addclassmetadata(type(self))

    def _addclassmetadata(self, cls):
        self._metadata['pdf_name'] = [cls.name]
        self._metadata['pdf_version'] = [cls.version]

    def _addmetadata(self, key, val):
        self._metadata[key] = np.expand_dims(val, 0)

    def _addobjdata(self, key, val):
        self._objdata[key] = val

    @property
    def metadata(self):
        """Return the metadata for this set of PDFs"""
        return self._metadata

    @property
    def objdata(self):
        """Return the object data for this set of PDFs"""
        return self._objdata

    @classmethod
    def create_gen(cls, **kwds):
        """Create and return a `scipy.stats.rv_continuous` object using the
        keyword arguemntets provided"""
        kwds_copy = kwds.copy()
        name = kwds_copy.pop('name', 'dist')
        return (cls(name=name), kwds_copy)

    @classmethod
    def create(cls, **kwds):
        """Create and return a `scipy.stats.rv_frozen` object using the
        keyword arguemntets provided"""
        # pylint: disable=not-callable
        obj, kwds_freeze = cls.create_gen(**kwds)
        return obj(**kwds_freeze)

    @classmethod
    def plot(cls, pdf, **kwargs):
        """Plot the pdf as a curve"""
        return plot_dist_pdf(pdf, **kwargs)

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        This defaults to plotting it as a curve, but this can be overwritten
        """
        return plot_dist_pdf(pdf, **kwargs)

    @classmethod
    def convert_from(cls, obj_from, method=None, **kwargs):
        """Convert a distribution or ensemble

        Parameters
        ----------
        obj_from :  `scipy.stats.rv_continuous or qp.ensemble`
            Input object
        method : `str`
            Optional argument to specify a non-default conversion algorithm
        kwargs : keyword arguments are passed to the output class constructor

        Returns
        -------
        ens : `qp.Ensemble`
            Ensemble of pdfs of this type using the data from obj_from
        """
        return qp_convert(obj_from, cls, method, **kwargs)


    def _moment_fix(self, n, *args, **kwds):
        """Hack fix for the moments calculation in scipy.stats, which can't handle
        the case of multiple PDFs.

        Parameters
        ----------
        n : int
            Order of the moment

        Returns
        -------
        moments : array_like
            The requested moments
        """
        # pylint: disable=no-member
        args, loc, scale = self._parse_args(*args, **kwds)
        cond = self._argcheck(*args) & (scale > 0)

        if np.floor(n) != n: #pragma: no cover
            raise ValueError("Moment must be an integer.")
        if n < 0: #pragma: no cover
            raise ValueError("Moment must be positive.")
        mu, mu2, g1, g2 = None, None, None, None
        if 0 < n < 5:
            if self._stats_has_moments:
                mdict = {'moments': {1: 'm', 2: 'v', 3: 'vs', 4: 'vk'}[n]}
            else:
                mdict = {}
            mu, mu2, g1, g2 = self._stats(*args, **mdict)
        val = np.where(cond, _moment_from_stats(n, mu, mu2, g1, g2, self._munp, args), np.nan)
        # Convert to transformed  X = L + S*Y
        # E[X^n] = E[(L+S*Y)^n] = L^n sum(comb(n, k)*(S/L)^k E[Y^k], k=0...n)
        def mom_at_zero():
            return scale**n * val
        def mom_non_zero():
            result = np.zeros(cond.shape)
            fac = scale / np.where(loc != 0, loc, 1)
            for k in range(n):
                valk = _moment_from_stats(k, mu, mu2, g1, g2, self._munp, args)
                result += comb(n, k, exact=True)*(fac**k) * valk
            result += fac**n * val
            return result * loc**n
        return np.where(loc==0, mom_at_zero(), mom_non_zero())



class rv_frozen_func(rv_frozen):
    """Trivial extention of `scipy.stats.rv_frozen`
    that includes the number of PDFs it represents
    """

    def __init__(self, dist, npdf, *args, **kwds):
        """C'tor

        Parameters
        ----------
        dist : `scipy.stats.rv_continuous`
            The underlying distribution
        npdf : `int`
            The number of PDFs this object represents
        """
        self._npdf = npdf
        super(rv_frozen_func, self).__init__(dist, *args, **kwds)

    @property
    def npdf(self):
        """Return the number of PDFs this object represents"""
        return self._npdf

    def histogramize(self, bins):
        """
        Computes integrated histogram bin values for all PDFs

        Parameters
        ----------
        bins: ndarray, float, optional
            Array of N+1 endpoints of N bins

        Returns
        -------
        self.histogram: ndarray, tuple, ndarray, floats
            Array of pairs of arrays of lengths (N+1, N) containing endpoints
            of bins and values in bins
        """
        cdf_vals = self.cdf(bins)
        bin_vals = cdf_vals[:,1:] - cdf_vals[:,0:-1]
        return (bins, bin_vals)



class Pdf_gen_simple(Pdf_gen):
    """Mixing class to extend `scipy.stats.rv_continuous` with
    information needed for `qp` for simple distributions.

    """
    def __init__(self, *args, **kwargs):
        """C'tor"""
        self._npdf = kwargs.get('npdf', 0)
        super(Pdf_gen_simple, self).__init__(*args, **kwargs)

    @property
    def npdf(self):
        """Return the number of PDFs this object represents"""
        return self._npdf

    def my_freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments.

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution.  Should include all
            the non-optional arguments, may include ``loc`` and ``scale``.

        Returns
        -------
        rv_frozen : rv_frozen instance
            The frozen distribution.
        """
        # pylint: disable=no-member
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (1, loc, scale))
        x = np.asarray((x - loc)/scale)
        args = tuple(map(asarray, args))
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        self._npdf = cond.shape[0]
        return rv_frozen_func(self, self._npdf, *args, **kwds)




class rv_frozen_rows(rv_frozen):
    """Trivial extention of `scipy.stats.rv_frozen`
    that includes to use when we want to have a collection
    of distribution of objects such as histograms or splines,
    where each object represents a single distribtuion
    """

    def __init__(self, dist, npdf, *args, **kwds):
        """C'tor"""
        self._npdf = npdf
        if self._npdf is not None:
            kwds.setdefault('row', np.expand_dims(np.arange(npdf), -1))
        super(rv_frozen_rows, self).__init__(dist, *args, **kwds)

    @property
    def npdf(self):
        """Return the number of PDFs this object represents"""
        return self._npdf


    def histogramize(self, bins):
        """
        Computes integrated histogram bin values for all PDFs

        Parameters
        ----------
        bins: ndarray, float, optional
            Array of N+1 endpoints of N bins

        Returns
        -------
        self.histogram: ndarray, tuple, ndarray, floats
            Array of pairs of arrays of lengths (N+1, N) containing endpoints
            of bins and values in bins
        """
        cdf_vals = self.cdf(bins)
        bin_vals = cdf_vals[:,1:] - cdf_vals[:,0:-1]
        return (bins, bin_vals)



class Pdf_rows_gen(rv_continuous, Pdf_gen):
    """Class extend `scipy.stats.rv_continuous` with
    information needed for `qp` when we want to have a collection
    of distribution of objects such as histograms or splines,
    where each object represents a single distribtuion

    """
    def __init__(self, *args, **kwargs):
        """C'tor"""
        self._npdf = kwargs.pop('npdf', 0)
        super(Pdf_rows_gen, self).__init__(*args, **kwargs)

    @property
    def npdf(self):
        """Return the number of PDFs this object represents"""
        return self._npdf

    @staticmethod
    def _sliceargs(x, row, *args):
        xx = np.unique(x)
        rr = np.unique(row)

        if np.size(xx) * np.size(rr) != np.size(x):
            return False, x, row, args
        outargs = [arg[0:np.size(xx)] for arg in args]
        return True, xx, rr, outargs


    def _argcheck(self, *args):
        """Default check for correct values on args and keywords.
        Returns condition array of 1's where arguments are correct and
         0's where they are not.
        """
        cond = 1
        if args:
            cond = np.logical_and(cond, np.logical_and(asarray(args[0]) >= 0, asarray(args[0]) < self._npdf))
        return np.atleast_1d(cond)

    def freeze(self, *args, **kwds):
        """Freeze the distribution for the given arguments.9999999

        Parameters
        ----------
        arg1, arg2, arg3,... : array_like
            The shape parameter(s) for the distribution.  Should include all
            the non-optional arguments, may include ``loc`` and ``scale``.

        Returns
        -------
        rv_frozen : rv_frozen instance
            The frozen distribution.
        """
        return rv_frozen_rows(self, self._npdf, *args, **kwds)

    @classmethod
    def create_gen(cls, **kwds):
        """Create and return a `scipy.stats.rv_continuous` object using the
        keyword arguemntets provided"""
        return (cls(**kwds), dict())

    def moment(self, n, *args, **kwds):
        """Returns the moments request moments for all the PDFs.
        This calls a hacked version `Pdf_gen._moment_fix` which can handle cases of multiple PDFs.

        Parameters
        ----------
        n : int
            Order of the moment

        Returns
        -------
        moments : array_like
            The requested moments
        """
        return Pdf_gen._moment_fix(self, n, *args, **kwds)
