"""This module implements a PDT distribution sub-class using a Gaussian mixture model
"""

from scipy.stats import rv_continuous
from scipy import integrate as sciint
import qp.sparse_rep as sparse_rep
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

    def __init__(self, sparse_indices, sparse_meta, *args, **kwargs):
        self._sparse_indices = sparse_indices
        self._sparse_meta = sparse_meta
        cut = kwargs.pop('cut', 1.e-5)
        #recreate the basis array from the metadata
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
        super(sparse2_gen, self).__init__(x, y.T, *args, **kwargs)
        self._xvals = x
        self._yvals = y.T
        for m in sparse_meta:
            self._metadata[m] = sparse_meta[m]

    def _updated_ctor_param(self):
        """
        Add the two constructor's arguments for the Factory
        """
        dct = super(sparse2_gen, self)._updated_ctor_param()
        dct['sparse_indices'] = self._sparse_indices
        dct['sparse_meta'] = self._sparse_meta
        return dct

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_sparse_from_xy, None)


sparse2 = sparse2_gen.create

add_class(sparse2_gen)
