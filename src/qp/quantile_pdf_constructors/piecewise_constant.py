from typing import List

import numpy as np

from qp.quantile_pdf_constructors.abstract_pdf_constructor import AbstractQuantilePdfConstructor
from qp.utils import evaluate_hist_multi_x_multi_y

class PiecewiseConstant(AbstractQuantilePdfConstructor):
    """This constructor takes the input quantiles and locations, and calculates a numerical
    derivative. We assume a constant value between derivative points and interpolate between
    those.
    """

    def __init__(self, quantiles, locations):
        """Constructor to instantiate this class.

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
        self._cdf_2nd_derivatives = None
        self._adjusted_locations = None

    def prepare_constructor(self):
        """This method will calculate the numerical derivative as well as the
        adjusted locations. The adjustments are necessary because the derivative is
        not a central derivative.
        """

        self._cdf_derivatives = np.zeros(self._locations.shape)
        self._cdf_2nd_derivatives = np.zeros(self._locations.shape)

        self._cdf_derivatives[:,0:-1] = (self._quantiles[1:] - self._quantiles[0:-1])/(self._locations[:,1:] - self._locations[:,0:-1])
        self._cdf_2nd_derivatives[:,0:-1] = self._cdf_derivatives[:,1:]  - self._cdf_derivatives[:,0:-1]

        # Offset the locations by -(l_[i+1] - l_i) / 2. So that the cdf_deriv can be correctly located.
        # This offset is necessary to correctly place the _cdf_derivs because we are using a
        # forward difference to calculate the numerical derivative.
        self._adjusted_locations = self._locations[:,1:]-np.diff(self._locations)/2


    def construct_pdf(self, grid, row: List[int] = None) -> List[List[float]]:
        """Take the intermediate calculations and return the interpolated y values
        given the input grid.

        Parameters
        ----------
        grid : List[float]
            List of x values to calculate y values for.
        row : List[int], optional
            A list of indexes of the original distribution to return, used as a filter.
            By default None will do no filtering.

        Returns
        -------
        List[List[float]]
            The lists of y values returned from self._interpolation_functions
        """
        if self._cdf_derivatives is None:  # pragma: no cover
            self.prepare_constructor()

        return evaluate_hist_multi_x_multi_y(grid, row, self._adjusted_locations,
            self._cdf_derivatives, self._cdf_2nd_derivatives).ravel()

    def debug(self):
        """Utility method to help with debugging. Returns input and intermediate
        calculations.

        Returns
        -------
            _quantiles :
                Input during constructor instantiation
            _locations :
                Input during constructor instantiation
        """
        return self._quantiles, self._locations, self._cdf_derivatives, self._cdf_2nd_derivatives, self._adjusted_locations