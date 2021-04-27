"""This module implements a PDT distribution sub-class using a Gaussian mixture model
"""

from scipy.stats import rv_continuous
from scipy import integrate as sciint
from qp import sparse_rep
from qp.factory import add_class
from qp.interp_pdf import interp_gen
from qp.conversion_funcs import extract_sparse_from_xy
from qp.test_data import SAMPLES, XARRAY, YARRAY, TEST_XVALS

class sparse_gen(interp_gen):
    """Sparse based distribution. The final behavior is similar to interp_gen, but the constructor
    takes a sparse representation to build the interpolator.
    Attempt to inherit from interp_gen : this is failing

    Notes
    -----
    This implements a qp interface to the original code SparsePz from M. Carrasco-Kind.

    """
    # pylint: disable=protected-access


    name = 'sparse'
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, xvals, mu, sig, N_SPARSE, dims, sparse_indices, *args, **kwargs):
        self.sparse_indices = sparse_indices
        self._xvals = xvals
        self.mu = mu
        self.sig = sig
        self.N_SPARSE = N_SPARSE
        self.dims = dims
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
        super(sparse_gen, self).__init__(*args, **kwargs)

        self._addmetadata('xvals', self._xvals)
        self._addmetadata('mu', self.mu)
        self._addmetadata('sig', self.sig)
        self._addmetadata('N_SPARSE', self.N_SPARSE)
        self._addmetadata('dims', self.dims)        
        self._addobjdata('sparse_indices', self.sparse_indices)
        #self._xvals = x
        #self._yvals = y.T
        #for m in sparse_meta:
        #    self._metadata[m] = sparse_meta[m]

    def _updated_ctor_param(self):
        """
        Add the two constructor's arguments for the Factory
        """
        dct = super(sparse_gen, self)._updated_ctor_param()
        dct['sparse_indices'] = self.sparse_indices
        dct['xvals'] = self._xvals
        dct['mu'] = self.mu
        dct['sig'] = self.sig
        dct['N_SPARSE'] = self.N_SPARSE
        dct['dims'] = self.dims
        return dct

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_sparse_from_xy, None)


sparse = sparse_gen.create

add_class(sparse_gen)

SPARSE_IDX, META, _ = sparse_rep.build_sparse_representation(XARRAY[-1], YARRAY)

sparse_gen.test_data = dict(sparse=dict(gen_func=sparse, \
                                        ctor_data=dict(xvals=META['xvals'], mu=META['mu'], sig=META['sig'],\
                                                       N_SPARSE=META['N_SPARSE'], dims=META['dims'], sparse_indices=SPARSE_IDX),\
                                        test_xvals=TEST_XVALS[::10]), )

