import numpy as np
import matplotlib.pyplot as plt

class PDF(object):

    def __init__(self, truth=None):
        self.truth = truth

    def evaluate(self, loc):#PDF

        return

    def integrate(self, limits):#CDF

        return

    def quantize(self, num_points):
        """
        Computes an array of evenly-spaced quantiles.

        Parameters
        ----------
        num_points : int
            The number of quantiles to compute.

        Returns
        -------
        self.quantiles : ndarray, float
            The quantile points.

        Comments
        --------
        Uses the `.ppf` method of the `rvs_continuous` distribution
        object stored in `self.truth`. This calculates the inverse CDF.
        See `the Scipy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.ppf.html#scipy.stats.rv_continuous.ppf>`_ for details.
        """
        # Find the spacing of the quantiles, and make an array of them:
        quantum = 1.0/float(num_points+1)
        points = np.linspace(0.0+quantum, 1.0-quantum, num_points)
        print("Calculating quantiles: ", points)
        self.quantiles = self.truth.ppf(points)
        print("Result: ", self.quantiles)
        return self.quantiles

    def interpolate(self):

        return

    def plot(self, limits, num_points):

        x = np.linspace(limits[0], limits[1], 100)

        y = [0., 1.]
        plt.plot(x, self.truth.pdf(x), color='k', linestyle='-', lw=1., alpha=1., label='true pdf')
        plt.vlines(self.quantize(num_points), y[0], y[1], color='k', linestyle='--', lw=1., alpha=1., label='quantiles')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('probability density')
        plt.savefig('plot.png')

        return
