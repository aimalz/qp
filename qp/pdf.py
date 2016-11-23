import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt

class PDF(object):

    def __init__(self, truth=None):
        self.truth = truth
        self.quantiles = None
        self.difs = None
        self.mids = None
        self.quantvals = None
        self.interpolator = None

    def evaluate(self, loc):

        return

    def integrate(self, limits):

        return

    def quantize(self, percent=1., number=None):
        """
        Computes an array of evenly-spaced quantiles.

        Parameters
        ----------
        percent : float
            The separation of the requested quantiles, in percent
        num_points : int
            The number of quantiles to compute.

        Returns
        -------
        self.quantiles : ndarray, float
            The quantile points.

        Comments
        --------
        Quantiles of a PDF could be a useful approximate way to store it. This method computes the quantiles, and stores them in the
        `self.quantiles` attribute.

        Uses the `.ppf` method of the `rvs_continuous` distribution
        object stored in `self.truth`. This calculates the inverse CDF.
        See `the Scipy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.ppf.html#scipy.stats.rv_continuous.ppf>`_ for details.
        """
        if number is not None:
            # Compute the spacing of the quantiles:
            quantum = 1.0 / float(number+1)
        else:
            quantum = percent/100.0
            # Over-write the number of quantiles:
            number = np.ceil(100.0 / percent) - 1
            assert number > 0

        points = np.linspace(0.0+quantum, 1.0-quantum, number)
        print("Calculating quantiles: ", points)
        self.quantiles = self.truth.ppf(points)
        self.difs = self.quantiles[1:]-self.quantiles[:-1]
        self.mids = (self.quantiles[1:]+self.quantiles[:-1])/2.
        self.quantvals = quantum/self.difs
        print("Result: ", self.quantiles)
        return self.quantiles

    def interpolate(self, number=100):
        """
        Interpolates the quantiles.

        Parameters
        ----------
        number: int
            The number of points over which to interpolate, bounded by the quantile value endpoints

        Returns
        -------
        grid: ndarray, float
            The input grid upon which to interpolate
        interpolated : ndarray, float
            The interpolated points.

        Notes
        -----

        """
        if self.interpolator is None:

            if self.quantiles is None:
                self.quantiles = self.quantize()

            print("Creating interpolator")
            self.interpolator = spi.interp1d(self.mids, self.quantvals)

        grid = np.linspace(min(self.mids), max(self.mids), number)
        print("Grid: ", grid)
        interpolated = self.interpolator(grid)
        print("Result: ", interpolated)

        return (grid, interpolated)

    def plot(self, limits, number=None):
        """
        Plot the PDF, in various ways.

        Parameters
        ----------
        limits : tuple, float
            Range over which to plot the PDF
        number: int
            Number of points over which to interpolate

        Notes
        -----
        What this method plots depends on what information about the PDF is stored in it: the more properties the PDF has, the more exciting the plot!
        """
        x = np.linspace(limits[0], limits[1], 100)

        if self.truth is not None:
            plt.plot(x, self.truth.pdf(x), color='k', linestyle='-', lw=1.0, alpha=1.0, label='True PDF')

        if self.quantiles is not None:
            y = [0., 1.]
            plt.vlines(self.quantiles, y[0], y[1], color='k', linestyle='--', lw=1.0, alpha=1., label='Quantiles')

        if number is not None:
            (grid, interpolated) = self.interpolate(number)
            plt.plot(grid, interpolated, color='r', linestyle=':', lw=2.0, alpha=1.0, label='Interpolated PDF')

        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Probability density')
        plt.savefig('plot.png')

        return
