"""This module implements a PDT distribution sub-class using histograms
"""


import numpy as np

from scipy.stats import rv_continuous

from qp.pdf_gen import Pdf_rows_gen
from qp.persistence import register_pdf_class
from qp.conversion import register_class_conversions
from qp.conversion_funcs import convert_using_hist_values, convert_using_hist_samples
from qp.plotting import get_axes_and_xlims, plot_pdf_histogram_on_axes
from qp.utils import interpolate_unfactored_multi_x_y, interpolate_unfactored_x_multi_y,\
     interpolate_multi_x_y, interpolate_x_multi_y




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

        check_input = kwargs.pop('check_input', True)
        if check_input:
            sums = np.sum(pdfs*self._hbin_widths, axis=1)
            self._hpdfs = (pdfs.T / sums).T
        else: #pragma: no cover
            self._hpdfs = pdfs
        self._hcdfs = None
        # Set support
        kwargs['a'] = self.a = self._hbins[0]
        kwargs['b'] = self.b = self._hbins[-1]
        kwargs['npdf'] = pdfs.shape[0]
        super(hist_rows_gen, self).__init__(*args, **kwargs)
        self._addmetadata('bins', self._hbins)
        self._addobjdata('pdfs', self._hpdfs)


    def _compute_cdfs(self):
        copy_shape = np.array(self._hpdfs.shape)
        copy_shape[-1] += 1
        self._hcdfs = np.ndarray(copy_shape)
        self._hcdfs[:,0] = 0.
        self._hcdfs[:,1:] = np.cumsum(self._hpdfs * self._hbin_widths, axis=1)



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
        if self._hcdfs is None: #pragma: no cover
            self._compute_cdfs()
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return interpolate_x_multi_y(xr, self._hbins, self._hcdfs[rr]).reshape(x.shape)
        return interpolate_unfactored_x_multi_y(xr, rr, self._hbins, self._hcdfs)

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        if self._hcdfs is None: #pragma: no cover
            self._compute_cdfs()
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

register_class_conversions(hist_rows_gen)
register_pdf_class(hist_rows_gen)
