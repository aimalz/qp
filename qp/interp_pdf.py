"""This module implements a PDT distribution sub-class using interpolated grids
"""

import numpy as np

from scipy.stats import rv_continuous

from qp.pdf_gen import Pdf_rows_gen
from qp.persistence import register_pdf_class
from qp.conversion import register_class_conversions
from qp.conversion_funcs import convert_using_vals_at_x, convert_using_xy_vals
from qp.plotting import get_axes_and_xlims, plot_pdf_on_axes
from qp.utils import normalize_interp1d,\
     interpolate_unfactored_multi_x_multi_y, interpolate_unfactored_multi_x_y, interpolate_unfactored_x_multi_y,\
     interpolate_multi_x_multi_y, interpolate_multi_x_y, interpolate_x_multi_y



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

        check_input = kwargs.pop('check_input', True)
        if check_input:
            self._compute_ycumul()
            self._yvals = (self._yvals.T / self._ycumul[:,-1]).T
            self._ycumul = (self._ycumul.T / self._ycumul[:,-1]).T
        else:  # pragma: no cover
            self._ycumul = None

        super(interp_fixed_grid_rows_gen, self).__init__(*args, **kwargs)
        self._addmetadata('xvals', self._xvals)
        self._addobjdata('yvals', self._yvals)

    def _compute_ycumul(self):
        copy_shape = np.array(self._yvals.shape)
        self._ycumul = np.ndarray(copy_shape)
        self._ycumul[:,0] = 0.
        self._ycumul[:,1:] = np.cumsum((self._xvals[1:]-self._xvals[0:-1])*self._yvals[:,1:], axis=1)

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
        if self._ycumul is None:  # pragma: no cover
            self._compute_ycumul()
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return interpolate_x_multi_y(xr, self._xvals, self._ycumul[rr]).reshape(x.shape)
        return interpolate_unfactored_x_multi_y(xr, rr, self._xvals, self._ycumul)

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if self._ycumul is None:  # pragma: no cover
            self._compute_ycumul()
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

        check_input = kwargs.pop('check_input', True)
        if check_input:
            self._yvals = normalize_interp1d(xvals, yvals)
        self._ycumul = None
        super(interp_rows_gen, self).__init__(*args, **kwargs)
        self._addobjdata('xvals', self._xvals)
        self._addobjdata('yvals', self._yvals)

    def _compute_ycumul(self):
        copy_shape = np.array(self._yvals.shape)
        self._ycumul = np.ndarray(copy_shape)
        self._ycumul[:,0] = 0.
        self._ycumul[:,1:] = np.cumsum(self._xvals[:,1:]*self._yvals[:,1:] - self._xvals[:,:-1]*self._yvals[:,1:], axis=1)


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
        if self._ycumul is None:  # pragma: no cover
            self._compute_ycumul()
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return interpolate_multi_x_multi_y(xr, self._xvals[rr], self._ycumul[rr], bounds_error=False, fill_value=(0., 1.)).reshape(x.shape)
        return interpolate_unfactored_multi_x_multi_y(xr, rr, self._xvals, self._ycumul, bounds_error=False, fill_value=(0., 1.))


    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        if self._ycumul is None:  # pragma: no cover
            self._compute_ycumul()
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


register_class_conversions(interp_rows_gen)
register_class_conversions(interp_fixed_grid_rows_gen)

register_pdf_class(interp_rows_gen)
register_pdf_class(interp_fixed_grid_rows_gen)
