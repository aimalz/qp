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
from scipy.stats import _continuous_distns as scipy_dist
from scipy.special import comb
from scipy import stats as sps

from scipy.interpolate import splev, splint

from .persistence import register_pdf_class
from .conversion import register_class_conversions, qp_convert
from .conversion_funcs import convert_using_vals_at_x, convert_using_xy_vals,\
     convert_using_quantiles, convert_using_samples, convert_using_hist_values,\
     convert_using_mixmod_fit_samples, convert_using_hist_samples
from .plotting import get_axes_and_xlims, plot_pdf_on_axes, plot_dist_pdf,\
     plot_pdf_histogram_on_axes, plot_pdf_quantiles_on_axes
from .utils import normalize_interp1d, normalize_spline, build_splines, build_kdes, evaluate_kdes,\
     interpolate_unfactored_multi_x_multi_y, interpolate_unfactored_multi_x_y, interpolate_unfactored_x_multi_y,\
     interpolate_multi_x_multi_y, interpolate_multi_x_y, interpolate_x_multi_y




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



class norm_gen(scipy_dist.norm_gen, Pdf_gen_simple):
    """Trival extension of the `scipy.stats.norm_gen` class for `qp`"""
    name = 'norm_dist'
    version = 0

    def __init__(self, *args, **kwargs):
        """C'tor"""
        npdf=None
        scipy_dist.norm_gen.__init__(self, *args, **kwargs)
        Pdf_gen_simple.__init__(self, npdf=npdf)

    def freeze(self, *args, **kwargs):
        """Overrides the freeze function to work with `qp`"""
        return self.my_freeze(*args, **kwargs)

    def _argcheck(self, *args):
        return np.atleast_1d(scipy_dist.norm_gen._argcheck(self, *args))

    def moment(self, n, *args, **kwds):
        """Returns the requested moments for all the PDFs.
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
        return self._moment_fix(n, *args, **kwds)

norm = norm_gen(name='norm_dist')


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


class hist_rows_gen(Pdf_rows_gen):
    """Histogram based distribution

    Notes
    -----
    This implements a PDF using a set of histogramed values.

    """
    # pylint: disable=protected-access

    name = 'hist_dist'
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, bins, pdfs, *args, **kwargs):
        """
        Create a new distribution using the given histogram
        Parameters
        ----------
        bins : array_like
          The second containing the (n+1) bin boundaries
        """
        self._hbins = np.asarray(bins)
        self._nbins = self._hbins.size - 1
        self._hbin_widths = self._hbins[1:] - self._hbins[:-1]
        if pdfs.shape[-1] != self._nbins: # pragma: no cover
            raise ValueError("Number of bins (%i) != number of values (%i)" % (self._nbins, pdfs.shape[-1]))
        sums = np.sum(pdfs*self._hbin_widths, axis=1)
        self._hpdfs = (pdfs.T / sums).T
        copy_shape = np.array(self._hpdfs.shape)
        copy_shape[-1] += 1
        self._hcdfs = np.ndarray(copy_shape)
        self._hcdfs[:,0] = 0.
        self._hcdfs[:,1:] = np.cumsum(self._hpdfs * self._hbin_widths, axis=1)
        # Set support
        kwargs['a'] = self.a = self._hbins[0]
        kwargs['b'] = self.b = self._hbins[-1]
        kwargs['npdf'] = pdfs.shape[0]
        super(hist_rows_gen, self).__init__(*args, **kwargs)
        self._addmetadata('bins', self._hbins)
        self._addobjdata('pdfs', self._hpdfs)

    @property
    def bins(self):
        """Return the histogram bin edges"""
        return self._hbins

    @property
    def pdfs(self):
        """Return the histogram bin values"""
        return self._hpdfs

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        idx = np.searchsorted(self._hbins, xr, side='left').clip(0, self._hbins.size-2)
        if factored:
            # x values and row values factorize
            return self._hpdfs[:,idx][rr].flat
        # x values and row values do not factorize, vectorize the call to histogram lookup
        def pdf_row(idxv, rv):
            return self._hpdfs[rv, idxv]
        vv = np.vectorize(pdf_row)
        return vv(idx, rr)


    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return interpolate_x_multi_y(xr, self._hbins, self._hcdfs[rr]).reshape(x.shape)
        return interpolate_unfactored_x_multi_y(xr, rr, self._hbins, self._hcdfs)

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return interpolate_multi_x_y(xr, self._hcdfs[rr], self._hbins).reshape(x.shape)
        return interpolate_unfactored_multi_x_y(xr, rr, self._hcdfs, self._hbins)

    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super(hist_rows_gen, self)._updated_ctor_param()
        dct['bins'] = self._hbins
        dct['pdfs'] = self._hpdfs
        return dct

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        For a histogram this shows the bin edges
        """
        axes, _, kw = get_axes_and_xlims(**kwargs)
        vals = pdf.dist.pdfs[pdf.kwds['row']]
        return plot_pdf_histogram_on_axes(axes, hist=(pdf.dist.bins, vals), **kw)


    @classmethod
    def add_conversion_mappings(cls, conv_dict):
        """
        Add this classes mappings to the conversion dictionary
        """
        conv_dict.add_mapping((cls.create, convert_using_hist_values), cls, None, None)
        conv_dict.add_mapping((cls.create, convert_using_hist_samples), cls, None, 'samples')



hist = hist_rows_gen.create



class interp_fixed_grid_rows_gen(Pdf_rows_gen):
    """Interpolator based distribution

    Notes
    -----
    This implements a PDF using a set of interpolated values.

    It simply takes a set of x and y values and uses `scipy.interpolate.interp1d` to
    build the PDF.
    """
    # pylint: disable=protected-access

    name = 'interp_fixed_dist'
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, xvals, yvals, *args, **kwargs):
        """
        Create a new distribution by interpolating the given values

        Parameters
        ----------
        xvals : array_like
          The x-values used to do the interpolation
        yvals : array_like
          The y-values used to do the interpolation
        """
        if xvals.size != np.sum(yvals.shape[1:]): # pragma: no cover
            raise ValueError("Shape of xbins in xvals (%s) != shape of xbins in yvals (%s)" % (xvals.size, np.sum(yvals.shape[1:])))
        self._xvals = xvals

        # Set support
        kwargs['a'] = self.a = np.min(self._xvals)
        kwargs['b'] = self.b = np.max(self._xvals)
        kwargs['npdf'] = yvals.shape[0]

        #self._yvals = normalize_interp1d(xvals, yvals)
        self._yvals = yvals
        copy_shape = np.array(self._yvals.shape)
        self._ycumul = np.ndarray(copy_shape)
        self._ycumul[:,0] = 0.
        self._ycumul[:,1:] = np.cumsum((self._xvals[1:]-self._xvals[0:-1])*self._yvals[:,1:], axis=1)

        self._yvals = (self._yvals.T / self._ycumul[:,-1]).T
        self._ycumul = (self._ycumul.T / self._ycumul[:,-1]).T

        super(interp_fixed_grid_rows_gen, self).__init__(*args, **kwargs)
        self._addmetadata('xvals', self._xvals)
        self._addobjdata('yvals', self._yvals)

    @property
    def xvals(self):
        """Return the x-values used to do the interpolation"""
        return self._xvals

    @property
    def yvals(self):
        """Return the y-valus used to do the interpolation"""
        return self._yvals

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return interpolate_x_multi_y(xr, self._xvals, self._yvals[rr]).reshape(x.shape)
        return interpolate_unfactored_x_multi_y(xr, rr, self._xvals, self._yvals)

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return interpolate_x_multi_y(xr, self._xvals, self._ycumul[rr]).reshape(x.shape)
        return interpolate_unfactored_x_multi_y(xr, rr, self._xvals, self._ycumul)

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return interpolate_multi_x_y(xr, self._ycumul[rr], self._xvals).reshape(x.shape)
        return interpolate_unfactored_multi_x_y(xr, rr, self._ycumul, self._xvals)

    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super(interp_fixed_grid_rows_gen, self)._updated_ctor_param()
        dct['xvals'] = self._xvals
        dct['yvals'] = self._yvals
        return dct

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        For a interpolated PDF this uses the interpolation points
        """
        axes, _, kw = get_axes_and_xlims(**kwargs)
        return plot_pdf_on_axes(axes, pdf, pdf.dist.xvals, **kw)

    @classmethod
    def add_conversion_mappings(cls, conv_dict):
        """
        Add this classes mappings to the conversion dictionary
        """
        conv_dict.add_mapping((cls.create, convert_using_vals_at_x), cls, None, None)


interp_fixed = interp_fixed_grid_rows_gen.create


class interp_rows_gen(Pdf_rows_gen):
    """Interpolator based distribution

    Notes
    -----
    This implements a PDF using a set of interpolated values.

    It simply takes a set of x and y values and uses `scipy.interpolate.interp1d` to
    build the PDF.
    """
    # pylint: disable=protected-access

    name = 'interp_dist'
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, xvals, yvals, *args, **kwargs):
        """
        Create a new distribution by interpolating the given values

        Parameters
        ----------
        xvals : array_like
          The x-values used to do the interpolation
        yvals : array_like
          The y-values used to do the interpolation
        """
        if xvals.shape != yvals.shape: # pragma: no cover
            raise ValueError("Shape of xvals (%s) != shape of yvals (%s)" % (xvals.shape, yvals.shape))
        self._xvals = xvals

        # Set support
        kwargs['a'] = self.a = np.min(self._xvals)
        kwargs['b'] = self.b = np.max(self._xvals)
        kwargs['npdf'] = xvals.shape[0]

        self._yvals = normalize_interp1d(xvals, yvals)
        copy_shape = np.array(self._yvals.shape)
        self._ycumul = np.ndarray(copy_shape)
        self._ycumul[:,0] = 0.
        self._ycumul[:,1:] = np.cumsum(self._xvals[:,1:]*self._yvals[:,1:] - self._xvals[:,:-1]*self._yvals[:,1:], axis=1)

        super(interp_rows_gen, self).__init__(*args, **kwargs)
        self._addobjdata('xvals', self._xvals)
        self._addobjdata('yvals', self._yvals)

    @property
    def xvals(self):
        """Return the x-values used to do the interpolation"""
        return self._xvals

    @property
    def yvals(self):
        """Return the y-valus used to do the interpolation"""
        return self._yvals

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return interpolate_multi_x_multi_y(xr, self._xvals[rr], self._yvals[rr], bounds_error=False, fill_value=0.).reshape(x.shape)
        return interpolate_unfactored_multi_x_multi_y(xr, rr, self._xvals, self._yvals, bounds_error=False, fill_value=0.)

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return interpolate_multi_x_multi_y(xr, self._xvals[rr], self._ycumul[rr], bounds_error=False, fill_value=(0., 1.)).reshape(x.shape)
        return interpolate_unfactored_multi_x_multi_y(xr, rr, self._xvals, self._ycumul, bounds_error=False, fill_value=(0., 1.))


    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return interpolate_multi_x_multi_y(xr, self._ycumul[rr], self._xvals[rr], bounds_error=False,
                                                   fill_value=(self.a, self.b)).reshape(x.shape)
        return interpolate_unfactored_multi_x_multi_y(xr, rr, self._ycumul, self._xvals, bounds_error=False,
                                                   fill_value=(self.a, self.b))



    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super(interp_rows_gen, self)._updated_ctor_param()
        dct['xvals'] = self._xvals
        dct['yvals'] = self._yvals
        return dct

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        For a interpolated PDF this uses the interpolation points
        """
        axes, _, kw = get_axes_and_xlims(**kwargs)
        xvals_row = pdf.dist.xvals
        return plot_pdf_on_axes(axes, pdf, xvals_row, **kw)

    @classmethod
    def add_conversion_mappings(cls, conv_dict):
        """
        Add this classes mappings to the conversion dictionary
        """
        conv_dict.add_mapping((cls.create, convert_using_xy_vals), cls, None, None)


interp = interp_rows_gen.create


class spline_rows_gen(Pdf_rows_gen):
    """Spline based distribution

    Notes
    -----
    This implements a PDF using a set splines
    """
    # pylint: disable=protected-access

    name = 'spline_dist'
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, *args, **kwargs):
        """
        Create a new distribution using the given histogram

        Keywords
        --------
        splx : array_like
          The x-values of the spline knots
        sply : array_like
          The y-values of the spline knots
        spln : array_like
          The order of the spline knots

        Notes
        -----
        Either (xvals and yvals) or (splx, sply and spln) must be provided.
        """
        splx = kwargs.pop('splx', None)
        sply = kwargs.pop('sply', None)
        spln = kwargs.pop('spln', None)

        if splx is None:  # pragma: no cover
            raise ValueError("Either splx must be provided")
        if splx.shape != sply.shape:  # pragma: no cover
            raise ValueError("Shape of xvals (%s) != shape of yvals (%s)" % (splx.shape, sply.shape))
        kwargs['a'] = self.a = np.min(splx)
        kwargs['b'] = self.b = np.max(splx)
        kwargs['npdf'] = splx.shape[0]
        self._splx = splx
        self._sply = sply
        self._spln = spln
        super(spline_rows_gen, self).__init__(*args, **kwargs)
        self._addobjdata('splx', self._splx)
        self._addobjdata('sply', self._sply)
        self._addobjdata('spln', self._spln)


    @staticmethod
    def build_normed_splines(xvals, yvals, **kwargs):
        """
        Build a set of normalized splines using the x and y values

        Parameters
        ----------
        xvals : array_like
          The x-values used to do the interpolation
        yvals : array_like
          The y-values used to do the interpolation

        Returns
        -------
        splx : array_like
          The x-values of the spline knots
        sply : array_like
          The y-values of the spline knots
        spln : array_like
          The order of the spline knots
        """
        if xvals.shape != yvals.shape:  # pragma: no cover
            raise ValueError("Shape of xvals (%s) != shape of yvals (%s)" % (xvals.shape, yvals.shape))
        xmin = np.min(xvals)
        xmax = np.max(xvals)
        yvals = normalize_spline(xvals, yvals, limits=(xmin, xmax), **kwargs)
        return build_splines(xvals, yvals)


    @classmethod
    def create_from_xy_vals(cls, xvals, yvals, **kwargs):
        """
        Create a new distribution using the given x and y values

        Parameters
        ----------
        xvals : array_like
          The x-values used to do the interpolation
        yvals : array_like
          The y-values used to do the interpolation

        Returns
        -------
        pdf_obj : `spline_rows_gen`
            The requested PDF
        """
        splx, sply, spln = spline_rows_gen.build_normed_splines(xvals, yvals, **kwargs)
        gen_obj = cls(splx=splx, sply=sply, spln=spln)
        return gen_obj(**kwargs)

    @classmethod
    def create_from_samples(cls, xvals, samples, **kwargs):
        """
        Create a new distribution using the given x and y values

        Parameters
        ----------
        xvals : array_like
          The x-values used to do the interpolation
        samples : array_like
          The sample values used to build the KDE

        Returns
        -------
        pdf_obj : `spline_rows_gen`
            The requested PDF
        """
        kdes = build_kdes(samples)
        kwargs.pop('yvals', None)
        yvals = evaluate_kdes(xvals, kdes)
        xvals_expand = (np.expand_dims(xvals, -1)*np.ones(samples.shape[0])).T
        return cls.create_from_xy_vals(xvals_expand, yvals, **kwargs)


    @property
    def splx(self):
        """Return x-values of the spline knots"""
        return self._splx

    @property
    def sply(self):
        """Return y-values of the spline knots"""
        return self._sply

    @property
    def spln(self):
        """Return order of the spline knots"""
        return self._spln

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        ns = self._splx.shape[-1]
        if factored:
            def pdf_row_fact(spl_):
                return splev(xr, (spl_[0:ns], spl_[ns:2*ns], spl_[-1].astype(int)))

            vv = np.vectorize(pdf_row_fact, signature="(%i)->(%i)" % (2*ns+1, xr.size))
            spl = np.hstack([self._splx[rr], self._sply[rr], self._spln[rr]])
            return vv(spl).flat

        def pdf_row(xv, irow):
            return splev(xv, (self._splx[irow], self._sply[irow], self._spln[irow]))

        vv = np.vectorize(pdf_row)
        return vv(xr, rr)


    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        def cdf_row(xv, irow):
            return splint(self.a, xv, (self._splx[irow], self._sply[irow], self._spln[irow]))

        vv = np.vectorize(cdf_row)
        return vv(x, row)

    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super(spline_rows_gen, self)._updated_ctor_param()
        dct['splx'] = self._splx
        dct['sply'] = self._sply
        dct['spln'] = self._spln
        return dct

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        For a spline this shows the spline knots
        """
        axes, _, kw = get_axes_and_xlims(**kwargs)
        xvals = pdf.dist.splx[pdf.kwds['row']]
        return plot_pdf_on_axes(axes, pdf, xvals, **kw)

    @classmethod
    def add_conversion_mappings(cls, conv_dict):
        """
        Add this classes mappings to the conversion dictionary
        """
        conv_dict.add_mapping((cls.create_from_xy_vals, convert_using_xy_vals), cls, None, None)
        conv_dict.add_mapping((cls.create_from_samples, convert_using_samples), cls, None, "samples")


spline = spline_rows_gen.create
spline_from_xy = spline_rows_gen.create_from_xy_vals
spline_from_samples = spline_rows_gen.create_from_samples


class quant_rows_gen(Pdf_rows_gen):
    """Spline based distribution

    Notes
    -----
    This implements a PDF by interpolating a set of quantile values

    It simply takes a set of x and y values and uses `scipy.interpolate.interp1d` to
    build the CDF
    """
    # pylint: disable=protected-access

    name = 'quant_dist'
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, quants, locs, *args, **kwargs):
        """
        Create a new distribution using the given values

        Parameters
        ----------
        quants : array_like
           The quantiles used to build the CDF
        locs : array_like
           The locations at which those quantiles are reached
        """
        kwargs['npdf'] = locs.shape[0]

        #kwargs['a'] = self.a = np.min(locs)
        #kwargs['b'] = self.b = np.max(locs)

        super(quant_rows_gen, self).__init__(*args, **kwargs)

        self._quants = np.asarray(quants)
        self._nquants = self._quants.size
        if locs.shape[-1] != self._nquants:  # pragma: no cover
            raise ValueError("Number of locations (%i) != number of quantile values (%i)" % (self._nquants, locs.shape[-1]))
        self._locs = locs

        self._addmetadata('quants', self._quants)
        self._addobjdata('locs', self._locs)

    @property
    def quants(self):
        """Return quantiles used to build the CDF"""
        return self._quants

    @property
    def locs(self):
        """Return the locations at which those quantiles are reached"""
        return self._locs

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return interpolate_multi_x_y(xr, self._locs[rr], self._quants, bounds_error=False, fill_value=(0., 1)).reshape(x.shape)
        return interpolate_unfactored_multi_x_y(xr, rr, self._locs, self._quants, bounds_error=False, fill_value=(0., 1))

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return interpolate_x_multi_y(xr, self._quants, self._locs[rr], bounds_error=False, fill_value=(self.a, self.b)).reshape(x.shape)
        return interpolate_unfactored_x_multi_y(xr, rr, self._quants, self._locs, bounds_error=False, fill_value=(self.a, self.b))

    def _updated_ctor_param(self):
        """
        Set the bins as additional construstor argument
        """
        dct = super(quant_rows_gen, self)._updated_ctor_param()
        dct['quants'] = self._quants
        dct['locs'] = self._locs
        return dct

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        For a quantile this shows the quantiles points
        """
        axes, xlim, kw = get_axes_and_xlims(**kwargs)
        xvals = np.linspace(xlim[0], xlim[1], kw.pop('npts', 101))
        locs = np.squeeze(pdf.dist.locs[pdf.kwds['row']])
        quants = np.squeeze(pdf.dist.quants)
        yvals = np.squeeze(pdf.pdf(xvals))
        return plot_pdf_quantiles_on_axes(axes, xvals, yvals, quantiles=(quants, locs), **kw)

    @classmethod
    def add_conversion_mappings(cls, conv_dict):
        """
        Add this classes mappings to the conversion dictionary
        """
        conv_dict.add_mapping((cls.create, convert_using_quantiles), cls, None, None)


quant = quant_rows_gen.create





class mixmod_rows_gen(Pdf_rows_gen):
    """Mixture model based distribution

    Notes
    -----
    This implements a PDF using a Gaussian Mixture model
    """
    # pylint: disable=protected-access

    name = 'mixmod_dist'
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, means, stds, weights, *args, **kwargs):
        """
        Create a new distribution using the given histogram

        Parameters
        ----------
        means : array_like
            The means of the Gaussians
        stds:  array_like
            The standard deviations of the Gaussians
        weights : array_like
            The weights to attach to the Gaussians
        """
        self._means = means
        self._stds = stds
        self._weights = weights
        kwargs['npdf'] = means.shape[0]
        self._ncomps = means.shape[1]
        super(mixmod_rows_gen, self).__init__(*args, **kwargs)
        self._addobjdata('weights', self._weights)
        self._addobjdata('stds', self._stds)
        self._addobjdata('means', self._means)

    @property
    def weights(self):
        """Return weights to attach to the Gaussians"""
        return self._weights

    @property
    def means(self):
        """Return means of the Gaussians"""
        return self._means

    @property
    def stds(self):
        """Return standard deviations of the Gaussians"""
        return self._stds

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return (np.expand_dims(self.weights[rr], -1) *\
                        sps.norm(loc=np.expand_dims(self._means[rr], -1),\
                                     scale=np.expand_dims(self._stds[rr], -1)).pdf(np.expand_dims(xr, 0))).sum(axis=1).reshape(x.shape)
        return (self.weights[rr].T * sps.norm(loc=self._means[rr].T, scale=self._stds[rr].T).pdf(xr)).sum(axis=0)
                                     
                                    
    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return (np.expand_dims(self.weights[rr], -1) *\
                        sps.norm(loc=np.expand_dims(self._means[rr], -1),\
                                    scale=np.expand_dims(self._stds[rr], -1)).cdf(np.expand_dims(xr, 0))).sum(axis=1).reshape(x.shape)
        return (self.weights[rr].T * sps.norm(loc=self._means[rr].T, scale=self._stds[rr].T).cdf(xr)).sum(axis=0)

                                    
    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super(mixmod_rows_gen, self)._updated_ctor_param()
        dct['means'] = self._means
        dct['stds'] = self._stds
        dct['weights'] = self._weights
        return dct

    @classmethod
    def add_conversion_mappings(cls, conv_dict):
        """
        Add this classes mappings to the conversion dictionary
        """
        conv_dict.add_mapping((cls.create, convert_using_mixmod_fit_samples), cls, None, None)


mixmod = mixmod_rows_gen.create


register_class_conversions(interp_rows_gen)
register_class_conversions(interp_fixed_grid_rows_gen)
register_class_conversions(spline_rows_gen)
register_class_conversions(quant_rows_gen)
register_class_conversions(hist_rows_gen)
register_class_conversions(mixmod_rows_gen)

register_pdf_class(norm_gen)
register_pdf_class(hist_rows_gen)
register_pdf_class(interp_rows_gen)
register_pdf_class(interp_fixed_grid_rows_gen)
register_pdf_class(spline_rows_gen)
register_pdf_class(quant_rows_gen)
register_pdf_class(mixmod_rows_gen)
