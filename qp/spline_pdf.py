"""This module implements a PDT distribution sub-class using splines
"""

import numpy as np

from scipy.stats import rv_continuous

from scipy.interpolate import splev, splint

from .pdf_gen import Pdf_rows_gen
from .persistence import register_pdf_class
from .conversion import register_class_conversions
from .conversion_funcs import convert_using_xy_vals, convert_using_samples
from .plotting import get_axes_and_xlims, plot_pdf_on_axes
from .utils import normalize_spline, build_splines, build_kdes, evaluate_kdes

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

register_class_conversions(spline_rows_gen)

register_pdf_class(spline_rows_gen)
