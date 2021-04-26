"""This module implements a PDT distribution sub-class using interpolated grids
"""

import numpy as np
from scipy.stats import rv_continuous
from scipy import integrate as sciint
import qp.sparse_rep as sparse_rep
from qp.factory import add_class
from qp.pdf_gen import Pdf_rows_gen
from qp.plotting import get_axes_and_xlims, plot_pdf_on_axes
from qp.conversion_funcs import extract_sparse_from_xy
from qp.utils import reshape_to_pdf_size, interpolate_x_multi_y, interpolate_unfactored_x_multi_y, interpolate_multi_x_y, interpolate_unfactored_multi_x_y

class sparse_gen(Pdf_rows_gen):
    """Sparse based distribution. The final behavior is similar to interp_gen, but the constructor
    takes a sparse representation to build the interpolator

    Notes
    -----
    This implements a qp interface to the original code SparsePz from M. Carrasco-Kind.

    """
    # pylint: disable=protected-access

    name = 'sparse'
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, xvals, mu, sig, N_SPARSE, dims, sparse_indices, *args, **kwargs):
        self._sparse_indices = sparse_indices
        self._xvals = xvals
        self._mu = mu
        self._sig = sig
        self._N_SPARSE = N_SPARSE
        self._dims = dims
        cut = kwargs.pop('cut', 1.e-5)
        #recreate the basis array from the metadata
        sparse_meta = dict(xvals=xvals, mu=mu, sig=sig, N_SPARSE=N_SPARSE, dims=dims)
        A = sparse_rep.create_basis(sparse_meta, cut=cut)
        #decode the sparse indices into basis indices and weights
        basis_indices, weights = sparse_rep.decode_sparse_indices(sparse_indices)
        #retrieve the weighted array of basis functions for each object
        pdf_y = A[:, basis_indices] * weights
        #normalize and sum the weighted pdfs
        x = sparse_meta['xvals']
        y = pdf_y.sum(axis=-1)
        norms = sciint.trapz(y.T, x)
        y /= norms
        yvals = y.T
        if x.size != np.sum(yvals.shape[1:]): # pragma: no cover
            raise ValueError("Shape of xvals (%s) != shape of yvals (%s)" % (x.size, np.sum(yvals.shape[1:])))
        self._xvals = x

        #the big basis matrix is not needed anymore
        del A

        #The following is boiler plate code from interp_pdf.py, but
        #it appears that sparse_pdf cannot inherit from interp_gen.
        #Something breaks when the Ensemble gets instantiated

        # Set support
        kwargs['a'] = self.a = np.min(self._xvals)
        kwargs['b'] = self.b = np.max(self._xvals)
        kwargs['shape'] = yvals.shape[:-1]

        #self._yvals = normalize_interp1d(xvals, yvals)
        self._yvals = reshape_to_pdf_size(yvals, -1)

        check_input = kwargs.pop('check_input', True)
        if check_input:
            self._compute_ycumul()
            self._yvals = (self._yvals.T / self._ycumul[:, -1]).T
            self._ycumul = (self._ycumul.T / self._ycumul[:, -1]).T
        else:  # pragma: no cover
            self._ycumul = None

        super(sparse_gen, self).__init__(*args, **kwargs)
        self._addmetadata('xvals', self._xvals)
        self._addmetadata('mu', self._mu)
        self._addmetadata('sig', self._sig)
        self._addmetadata('N_SPARSE', self._N_SPARSE)
        self._addmetadata('dims', self._dims)        
        self._addobjdata('sparse_indices', self._sparse_indices)
        
    @property
    def sparse_indices(self):
        """Return sparse_indices as property"""
        return self._sparse_indices

    def _compute_ycumul(self):
        copy_shape = np.array(self._yvals.shape)
        self._ycumul = np.ndarray(copy_shape)
        self._ycumul[:, 0] = 0.5 * self._yvals[:, 0] * (self._xvals[1] - self._xvals[0])
        self._ycumul[:, 1:] = np.cumsum((self._xvals[1:] - self._xvals[:-1]) *
                                        0.5 * np.add(self._yvals[:, 1:],
                                                     self._yvals[:, :-1]), axis=1)

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
            return interpolate_x_multi_y(xr, self._xvals, self._yvals[rr], bounds_error=False,
                                         fill_value=0.).reshape(x.shape)
        return interpolate_unfactored_x_multi_y(xr, rr, self._xvals, self._yvals,
                                                bounds_error=False, fill_value=0.)

    def _cdf(self, x, row):
        # pylint: disable=arguments-differ
        if self._ycumul is None:  # pragma: no cover
            self._compute_ycumul()
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            return interpolate_x_multi_y(xr, self._xvals, self._ycumul[rr],
                                         bounds_error=False, fill_value=(0., 1.)).reshape(x.shape)
        return interpolate_unfactored_x_multi_y(xr, rr, self._xvals, self._ycumul,
                                                bounds_error=False, fill_value=(0., 1.))

    def _ppf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if self._ycumul is None:  # pragma: no cover
            self._compute_ycumul()
        if factored:
            return interpolate_multi_x_y(xr, self._ycumul[rr], self._xvals, bounds_error=False,
                                         fill_value=(0., 1.)).reshape(x.shape)
        return interpolate_unfactored_multi_x_y(xr, rr, self._ycumul, self._xvals,
                                                bounds_error=False, fill_value=(0., 1.))

    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super(sparse_gen, self)._updated_ctor_param()
        dct['sparse_indices'] = self._sparse_indices
        dct['xvals'] = self._xvals
        dct['mu'] = self._mu
        dct['sig'] = self._sig
        dct['N_SPARSE'] = self._N_SPARSE
        dct['dims'] = self._dims
        return dct

    @classmethod
    def plot_native(cls, pdf, **kwargs):
        """Plot the PDF in a way that is particular to this type of distibution

        For a interpolated PDF this uses the interpolation points
        """
        axes, _, kw = get_axes_and_xlims(**kwargs)
        return plot_pdf_on_axes(axes, pdf, pdf.dist.xvals, **kw)

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_sparse_from_xy, None)


sparse = sparse_gen.create

add_class(sparse_gen)
