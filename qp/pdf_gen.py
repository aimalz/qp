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
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats import _continuous_distns as scipy_dist
from scipy import stats as sps

from scipy.interpolate import interp1d, splev, splint

from .persistence import register_pdf_class
from .conversion import register_class_conversions, qp_convert
from .conversion_funcs import convert_using_xy_vals,\
     convert_using_quantiles, convert_using_samples, convert_using_hist_values, convert_using_hist_samples,\
     convert_using_mixmod_fit_samples
from .plotting import get_axes_and_xlims, plot_pdf_on_axes, plot_dist_pdf,\
     plot_pdf_histogram_on_axes, plot_pdf_quantiles_on_axes, plot_pdf_samples_on_axes
from .utils import normalize_interp1d, normalize_spline, build_splines, build_kdes, evaluate_kdes




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

    conversion_map = {}

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


    def _argcheck(self, *args):
        """Default check for correct values on args and keywords.
        Returns condition array of 1's where arguments are correct and
         0's where they are not.
        """
        cond = 1
        if args:
            cond = np.logical_and(cond, np.logical_and(asarray(args[0]) >= 0, asarray(args[0]) < self._npdf))
        return cond

    def freeze(self, *args, **kwds):
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
        return rv_frozen_rows(self, self._npdf, *args, **kwds)

    @classmethod
    def create_gen(cls, **kwds):
        """Create and return a `scipy.stats.rv_continuous` object using the
        keyword arguemntets provided"""
        return (cls(**kwds), dict())


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

    conversion_map = {None:convert_using_hist_values}

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
        def pdf_row(irow, xv):
            return self._hpdfs[irow,np.searchsorted(self._hbins, xv, side='left').clip(0, self._nbins-1)]
        vv = np.vectorize(pdf_row)
        return vv(row, x)

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        def cdf_row(irow, xv):
            return interp1d(self._hbins, self._hcdfs[irow], bounds_error=False,  fill_value=(0,1.))(xv)
        vv = np.vectorize(cdf_row)
        return vv(row, x)

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        def ppf_row(irow, xv):
            return interp1d(self._hcdfs[irow], self._hbins, bounds_error=False,  fill_value=(self.a, self.b))(xv)
        vv = np.vectorize(ppf_row)
        return vv(row, x)

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




hist = hist_rows_gen.create


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

    conversion_map = {None:convert_using_xy_vals}

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
        def pdf_row(irow, xv):
            return interp1d(self._xvals[irow], self._yvals[irow])(xv)

        vv = np.vectorize(pdf_row)
        return vv(row, x)

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        def cdf_row(irow, xv):
            return interp1d(self._xvals[irow], self._ycumul[irow])(xv)
        vv = np.vectorize(cdf_row)
        return vv(row, x)

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        def ppf_row(irow, xv):
            return interp1d(self._ycumul[irow], self._xvals[irow])(xv)
        vv = np.vectorize(ppf_row)
        return vv(row, x)

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
        xvals_row = pdf.dist.xvals[pdf.kwds['row']]
        return plot_pdf_on_axes(axes, pdf, xvals_row, **kw)


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

    conversion_map = {None:convert_using_xy_vals}

    def __init__(self, *args, **kwargs):
        """
        Create a new distribution using the given histogram

        Keywords
        --------
        xvals : array_like
          The x-values used to do the interpolation
        yvals : array_like
          The y-values used to do the interpolation
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
        xvals = kwargs.pop('xvals', None)
        yvals = kwargs.pop('yvals', None)
        splx = kwargs.pop('splx', None)
        sply = kwargs.pop('sply', None)
        spln = kwargs.pop('spln', None)

        if xvals is None:
            if splx is None:  # pragma: no cover
                raise ValueError("Either xvals or splx must be provided")
            if splx.shape != sply.shape:  # pragma: no cover
                raise ValueError("Shape of xvals (%s) != shape of yvals (%s)" % (splx.shape, sply.shape))
            kwargs['a'] = self.a = np.min(splx)
            kwargs['b'] = self.b = np.max(splx)
            kwargs['npdf'] = splx.shape[0]
            self._xvals = None
            self._yvals = None
            self._splx = splx
            self._sply = sply
            self._spln = spln
        else:
            if splx is not None:  # pragma: no cover
                raise ValueError("Only one of yvals or splreps must be provided")
            if xvals.shape != yvals.shape:  # pragma: no cover
                raise ValueError("Shape of xvals (%s) != shape of yvals (%s)" % (xvals.shape, yvals.shape))
            self._xvals = xvals
            self.a = np.min(xvals)
            self.b = np.max(xvals)
            self._yvals = normalize_spline(xvals, yvals, limits=(self.a, self.b), **kwargs)
            self._splx, self._sply, self._spln = build_splines(self._xvals, self._yvals)
            kwargs['a'] = self.a
            kwargs['b'] = self.b
            kwargs['npdf'] = xvals.shape[0]

        super(spline_rows_gen, self).__init__(*args, **kwargs)
        self._addobjdata('splx', self._splx)
        self._addobjdata('sply', self._sply)
        self._addobjdata('spln', self._spln)

    @property
    def xvals(self):
        """Return x-values used to do the interpolation"""
        return self._xvals

    @property
    def yvals(self):
        """Return y-values used to do the interpolation"""
        return self._yvals

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
        def pdf_row(irow, xv):
            return splev(xv, (self._splx[irow], self._sply[irow], self._spln[irow]))

        vv = np.vectorize(pdf_row)
        return vv(row, x)

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        def cdf_row(irow, xv):
            return splint(self.a, xv, (self._splx[irow], self._sply[irow], self._spln[irow]))

        vv = np.vectorize(cdf_row)
        return vv(row, x)

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


spline = spline_rows_gen.create


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

    conversion_map = {None:convert_using_quantiles}


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
        def cdf_row(irow, xv):
            return interp1d(self._locs[irow], self._quants, bounds_error=False,  fill_value=(0,1.))(xv)
        vv = np.vectorize(cdf_row)
        return vv(row, x)

    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
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



quant = quant_rows_gen.create



class kde_rows_gen(Pdf_rows_gen):
    """Kernal density estimate based distribution

    Notes
    -----
    This implements a PDF using a Kernal density estimate

    It simply takes a set of samples and uses `scipy.stats.gaussian_kde`
    to build the PDF.
    """
    # pylint: disable=protected-access

    name = 'kde_dist'
    version = 0

    _support_mask = rv_continuous._support_mask

    conversion_map = {None:convert_using_samples}

    def __init__(self, xvals, *args, **kwargs):
        """
        Create a new distribution using the given samples

        Parameters
        ----------
        xvals : array_like
            X-values at which to define the distribution

        Keywords
        --------
        samples : array_like
            Samples used to build the KDE
        yvals : array_like
            y-value constructed from the KDE
        """
        self._samples = kwargs.pop('samples', None)
        self._xvals = xvals
        self._yvals = kwargs.pop('yvals', None)
        if self._samples is not None:
            self._kdes = build_kdes(self._samples)
            self._yvals = evaluate_kdes(self._xvals, self._kdes)
        else:
            if self._yvals is None: #pragma: no cover
                raise ValueError("Either samples or yvals must be specified")
            self._kdes = None
        copy_shape = np.array(self._yvals.shape)
        self._ycumul = np.ndarray(copy_shape)
        self._ycumul[:,0] = 0.
        self._ycumul[:,1:] = np.cumsum(self._xvals[1:]*self._yvals[:,1:] - self._xvals[:-1]*self._yvals[:,1:], axis=1)

        kwargs['npdf'] = self._yvals.shape[0]
        super(kde_rows_gen, self).__init__(*args, **kwargs)
        self._addmetadata('xvals', self._xvals)
        self._addobjdata('yvals', self._yvals)


    @property
    def samples(self):
        """Return the samples used to build the KDEs"""
        return self._samples

    @property
    def kdes(self):
        """Return the KDEs"""
        return self._kdes

    @property
    def xvals(self):
        """Return the xvalues"""
        return self._xvals

    @property
    def yvals(self):
        """Return the xvalues"""
        return self._yvals


    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        def pdf_row(irow, xv):
            return interp1d(self._xvals, self._yvals[irow])(xv)

        vv = np.vectorize(pdf_row)
        return vv(row, x)

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        def cdf_row(irow, xv):
            return interp1d(self._xvals, self._ycumul[irow])(xv)
        vv = np.vectorize(cdf_row)
        return vv(row, x)

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        def ppf_row(irow, xv):
            return interp1d(self._ycumul[irow], self._xvals)(xv)
        vv = np.vectorize(ppf_row)
        return vv(row, x)


    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super(kde_rows_gen, self)._updated_ctor_param()
        dct['samples'] = self._samples
        dct['xvals'] = self._xvals
        dct['yvals'] = self._yvals
        return dct


    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        For a kde this shows the samples
        """
        axes, _, kw = get_axes_and_xlims(**kwargs)
        if pdf.dist.samples is None: #pragma: no cover
            xvals_row = pdf.dist.xvals[pdf.kwds['row']]
            return plot_pdf_on_axes(axes, pdf, xvals_row, **kw)
        samples = pdf.dist.samples[pdf.kwds['row']]
        return plot_pdf_samples_on_axes(axes, pdf, samples, **kw)



kde = kde_rows_gen.create



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

    conversion_map = {None:convert_using_mixmod_fit_samples}

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
        return self._weights[row].T * sps.norm(loc=self._means[row].T, scale=self._stds[row].T).pdf(x).sum(axis=0)

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        return self._weights[row].T * sps.norm(loc=self._means[row].T, scale=self._stds[row].T).cdf(x).sum(axis=0)

    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super(mixmod_rows_gen, self)._updated_ctor_param()
        dct['means'] = self._means
        dct['stds'] = self._stds
        dct['weights'] = self._weights
        return dct

mixmod = mixmod_rows_gen.create


hist_rows_gen.conversion_map[kde_rows_gen] = convert_using_hist_samples


register_class_conversions(interp_rows_gen)
register_class_conversions(spline_rows_gen)
register_class_conversions(quant_rows_gen)
register_class_conversions(kde_rows_gen)
register_class_conversions(hist_rows_gen)
register_class_conversions(mixmod_rows_gen)

register_pdf_class(norm_gen)
register_pdf_class(hist_rows_gen)
register_pdf_class(interp_rows_gen)
register_pdf_class(spline_rows_gen)
register_pdf_class(kde_rows_gen)
register_pdf_class(quant_rows_gen)
register_pdf_class(mixmod_rows_gen)
