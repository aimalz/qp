"""This module implements a PDT distribution sub-class using a Gaussian mixture model
"""

from scipy.stats import rv_continuous
from scipy import integrate as sciint
from qp import sparse_rep
from qp.factory import add_class
from qp.interp_pdf import interp_gen
from qp.conversion_funcs import extract_sparse_from_xy

class sparse2_gen(interp_gen):
    """Sparse based distribution. The final behavior is similar to interp_gen, but the constructor
    takes a sparse representation to build the interpolator.
    Attempt to inherit from interp_gen : this is failing

    Notes
    -----
    This implements a qp interface to the original code SparsePz from M. Carrasco-Kind.

    """
    # pylint: disable=protected-access


    name = 'sparse2'
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
        kwargs.setdefault('xvals', x)
        kwargs.setdefault('yvals', y.T)
        super(sparse2_gen, self).__init__(*args, **kwargs)

        self._addmetadata('xvals', self._xvals)
        self._addmetadata('mu', self._mu)
        self._addmetadata('sig', self._sig)
        self._addmetadata('N_SPARSE', self._N_SPARSE)
        self._addmetadata('dims', self._dims)        
        self._addobjdata('sparse_indices', self._sparse_indices)
        #self._xvals = x
        #self._yvals = y.T
        #for m in sparse_meta:
        #    self._metadata[m] = sparse_meta[m]

    def _updated_ctor_param(self):
        """
        Add the two constructor's arguments for the Factory
        """
        dct = super(sparse2_gen, self)._updated_ctor_param()
        dct['sparse_indices'] = self._sparse_indices
        dct['xvals'] = self._xvals
        dct['mu'] = self._mu
        dct['sig'] = self._sig
        dct['N_SPARSE'] = self._N_SPARSE
        dct['dims'] = self._dims
        return dct

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_sparse_from_xy, None)


sparse2 = sparse2_gen.create

sparse2_gen.test_data = {}

add_class(sparse2_gen)
