"""Module to define qp distributions that inherit from scipy distributions"""

import numpy as np

from scipy.stats import _continuous_distns as scipy_dist

from .pdf_gen import Pdf_gen_simple
from .persistence import register_pdf_class

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

register_pdf_class(norm_gen)
