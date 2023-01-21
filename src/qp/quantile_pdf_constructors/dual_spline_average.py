from typing import List, Optional

import numpy as np
from scipy.interpolate import interp1d

from qp.quantile_pdf_constructors.abstract_pdf_constructor import AbstractQuantilePdfConstructor

class DualSplineAverage(AbstractQuantilePdfConstructor):
    """Drew's implementation of the "area-under-the-curve" using the average of
    the bounding splines fit to the CDF derivative.
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
        self._p_of_zs = np.zeros(self._locations.shape)

    def prepare_constructor(self):
        first_term = np.squeeze(2 * np.diff(self._quantiles) / np.diff(self._locations))
        for i in range(1, np.shape(self._p_of_zs)[-1]):
            self._p_of_zs[0,i] = max(0.0, first_term[i-1] - self._p_of_zs[0,i-1])

    def construct_pdf(self, grid, row) -> List[List[float]]:
        f1 = interp1d(np.squeeze(self._locations[0,0::2]), np.squeeze(self._p_of_zs[0,0::2]), bounds_error=False, fill_value=0.0, kind='cubic')
        f2 = interp1d(np.squeeze(self._locations[0,1::2]), np.squeeze(self._p_of_zs[0,1::2]), bounds_error=False, fill_value=0.0, kind='cubic')

        y1 = f1(grid)
        y2 = f2(grid)

        return (y1 + y2) / 2
