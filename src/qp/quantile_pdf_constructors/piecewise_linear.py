from typing import List, Optional

import numpy as np

from qp.quantile_pdf_constructors.abstract_pdf_constructor import AbstractQuantilePdfConstructor
from qp.utils import interpolate_multi_x_multi_y

class PiecewiseLinear(AbstractQuantilePdfConstructor):
    """_summary_
    """

    def __init__(self, quantiles, locations):
        """_summary_

        Parameters
        ----------
        quantiles : List[float]
            List of n quantile values in the range (0,1).
        locations : List[List[float]]
            List of m Lists, each containing n values corresponding to the
            y-value of the PPF function at the same quantile index.
        """
        self._quantiles = quantiles
        self._locations = locations

        self._cdf_derivatives = None
        self._adjusted_locations = None

    def prepare_constructor(self):
        # calculate the first derivative using a forward difference.
        self._cdf_derivatives = (self._quantiles[1:] - self._quantiles[0:-1])/(self._locations[:,1:] - self._locations[:,0:-1])

        # Offset the locations by -(l_[i+1] - l_i) / 2. So that the cdf_deriv can be correctly located.
        # This offset is necessary to correctly place the _cdf_derivs because we are using a
        # forward difference to calculate the numerical derivative.
        self._adjusted_locations = self._locations[:,1:]-np.diff(self._locations)/2

    def construct_pdf(self, grid, row) -> List[List[float]]:
        if self._cdf_derivatives is None:  # pragma: no cover
            self.prepare_constructor()

        return interpolate_multi_x_multi_y(grid, row, self._adjusted_locations, self._cdf_derivatives,
            bounds_error=False, fill_value=(0.,0.), kind='linear').ravel()
