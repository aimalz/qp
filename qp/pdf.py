import numpy as np
import scipy.interpolate as spi
import matplotlib.pyplot as plt

import qp

class PDF(object):

    def __init__(self, truth=None, quantiles=None, histogram=None):
        """
        Initializes the PDF object with some representation of a distribution.

        Parameters
        ----------
        truth: scipy.stats.rv_continuous object, optional
            Continuous, parametric form of the PDF
        quantiles: ndarray, optional
            Array of quantile values separated by 100./(len(quantiles)+1) percentiles
        histogram: tuple of ndarrays, optional
            Pair of arrays of lengths (nbins+1, nbins) containing endpoints of bins and values in bins
        """
        self.truth = truth
        self.quantiles = quantiles
        self.histogram = histogram
        self.initialized = None

        if self.truth is not None:
            self.initialized = 'truth'
        elif self.quantiles is not None:
            self.initialized = 'quantiles'
        elif self.histogram is not None:
            self.initialized = 'histogram'
        self.last = self.initialized

        # Should make this a proper exception rather than just printing an advisory notice
        if self.truth is None and self.quantiles is None and self.histogram is None:
            print('It is unwise to initialize a PDF object without inputs!')
            return
        self.interpolator = None

    def evaluate(self, loc, vb=True):
        """
        Evaluates the PDF at given location(s) using truth if available and last-calculated approximation otherwise.

        Parameters
        ----------
        loc: float or ndarray
            Location(s) at which to evaluate the pdf
        vb: boolean
            Report on progress to stdout?

        Returns
        -------
        val: float or ndarray
            Value of the truth function at given location(s)

        Comments
        --------
        This function evaluates the truth function if it is available and the interpolated quantile approximation otherwise.
        """
        if self.truth is not None:
            if vb: print('Evaluating the true distribution.')
            val = self.truth.pdf(loc)
        else:
            if vb: print('Evaluating an interpolation of the '+self.last+' parametrization.')
            val = self.approximate(loc, using=self.last)[1]

        return(val)

    def integrate(self, limits):

        return

    def quantize(self, percent=1., number=None, vb=True):
        """
        Computes an array of evenly-spaced quantiles from the truth.

        Parameters
        ----------
        percent : float
            The separation of the requested quantiles, in percent
        number : int
            The number of quantiles to compute.
        vb: boolean
            Report on progress to stdout?

        Returns
        -------
        self.quantiles : ndarray, float
            The quantile points.

        Comments
        --------
        Quantiles of a PDF could be a useful approximate way to store it. This method computes the quantiles from a truth distribution (other representations forthcoming)
        and stores them in the `self.quantiles` attribute.

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
        if vb: print("Calculating quantiles: ", points)
        if self.truth is not None:
            self.quantiles = self.truth.ppf(points)
        else:
            print('New quantiles can only be computed from a truth distribution in this version.')
            return

        if vb: print("Result: ", self.quantiles)
        self.last = 'quantiles'
        return self.quantiles

    def histogramize(self, binends=None, nbins=10, binrange=[0., 1.], vb=True):
        """
        Computes the histogram values from the truth.

        Parameters
        ----------
        binends: ndarray, float, optional
            Array of N+1 endpoints of N bins
        nbins: int, optional
            Number of bins if no binends provided
        range: tuple, float, optional
            Pair of values of endpoints of total bin range
        vb: boolean
            Report on progress to stdout?

        Returns
        -------
        self.histogram: tuple of ndarrays of floats
            Pair of arrays of lengths (nbins+1, nbins) containing endpoints of bins and values in bins

        Comments
        --------
        A histogram representation of a PDF is a popular approximate way to store it. This method computes some histogram bin heights from a truth distribution (other representations forthcoming)
        and stores them in the `self.histogram` attribute.

        Uses the `.cdf` method of the `rvs_continuous` distribution
        object stored in `self.truth`. This calculates the CDF.
        See `the Scipy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.cdf.html#scipy.stats.rv_continuous.cdf>`_ for details.

        """
        if binends is None:
            step = float(binrange[1]-binrange[0])/nbins
            binends = np.arange(binrange[0], binrange[1]+step, step)

        nbins = len(binends)-1
        self.histogram = np.zeros(nbins)
        if vb: print("Calculating histogram: ", binends)
        if self.truth is not None:
            cdf = self.truth.cdf(binends)
            for b in range(nbins):
                self.histogram[b] = (cdf[b+1]-cdf[b])/(binends[b+1]-binends[b])
        else:
            print('New histograms can only be computed from a truth distribution in this version.')
            return

        self.histogram = (binends, self.histogram)
        if vb: print("Result: ", self.histogram[1])
        self.last = 'histogram'
        return self.histogram

    def interpolate(self, using=self.last, vb=True):
        """
        Constructs an `interpolator` function based on the parametrization.

        Parameters
        ----------
        using: string
            Parametrization on which to interpolate, currently supports 'quantiles', 'histogram'
        vb: boolean
            Report on progress to stdout?

        Returns
        -------
        None

        Notes
        -----
        The `self.interpolator` object is a function that is used by the `approximate` method.
        """
        if using == 'truth':
            print('The truth needs no interpolation.  Try converting to an approximate parametrization first.')
            return

        if using == 'quantiles':
            # First find the quantiles if none exist:
            if self.quantiles is None:
                self.quantiles = self.quantize()

            self.difs = self.quantiles[1:]-self.quantiles[:-1]
            self.mids = (self.quantiles[1:]+self.quantiles[:-1])/2.
            self.vals = (1.0/(len(self.quantiles)+1))/self.difs

        if using == 'histogram':
            # First find the histogram if none exists:
            if self.histogram is None:
                self.histogram = self.histogramize()

            self.mids = (self.histogram[0][1:]+self.histogram[0][:-1])/2.
            self.vals = self.histogram[1]

        if vb: print("Creating interpolator")
        self.interpolator = spi.interp1d(self.mids, self.vals, fill_value="extrapolate")

        return

    def approximate(self, points, using, vb=True):
        """
        Interpolates the parametrization to get an approximation to the density.

        Parameters
        ----------
        number: int
            The number of points over which to interpolate, bounded by the quantile value endpoints
        points: ndarray
            The value(s) at which to evaluate the interpolated function
        using: string
            Parametrization from which to approximate, currently supports 'quantiles', 'histogram'
        vb: boolean
            Report on progress to stdout?

        Returns
        -------
        points: ndarray, float
            The input grid upon which to interpolate
        interpolated : ndarray, float
            The interpolated points.

        Comments
        --------
        Extrapolation is linear while values are positive; otherwise, extrapolation returns 0.

        Notes
        -----
        Example:
            x, y = p.approximate(np.linspace(-1., 1., 100))
        """

        if self.interpolator is None:
            self.interpolate(using=using)
        interpolated = self.interpolator(points)
        interpolated[interpolated<0.] = 0.

        return (points, interpolated)

    def plot(self, limits, points=None):
        """
        Plot the PDF, in various ways.

        Parameters
        ----------
        limits : tuple, float
            Range over which to plot the PDF
        points: ndarray, optional
            The value(s) at which to evaluate the interpolator

        Notes
        -----
        What this method plots depends on what information about the PDF is stored in it: the more properties the PDF has, the more exciting the plot!
        """

        x = np.linspace(limits[0], limits[1], 100)

        if self.truth is not None:
            plt.plot(x, self.truth.pdf(x), color='k', linestyle='-', lw=1.0, alpha=1.0, label='True PDF')

        if self.quantiles is not None:
            plt.vlines(self.quantiles, 0., 1., color='b', linestyle='--', lw=1.0, alpha=1., label='Quantiles')
            if points is not None:
                (grid, qinterpolated) = self.approximate(points, using='quantiles')
                plt.plot(grid, qinterpolated, color='b', linestyle=':', lw=2.0, alpha=1.0, label='Quantile Interpolated PDF')

        if self.histogram is not None:
            plt.hlines(self.histogram[1], self.histogram[0][:-1], self.histogram[0][1:], color='r', linestyle='--', lw=1.0, alpha=1., label='Histogram')
            if points is not None:
                (grid, hinterpolated) = self.approximate(points, using='histogram')
                plt.plot(grid, hinterpolated, color='r', linestyle=':', lw=2.0, alpha=1.0, label='Histogram Interpolated PDF')

        plt.legend()
        plt.xlabel('x')
        plt.ylabel('Probability density')
        plt.savefig('plot.png')

        return

    def kld(self, limits=(0., 1.), dx=0.01):
        """
        Calculates Kullback-Leibler divergence of quantile approximation from truth.

        Parameters
        ----------
        limits: tuple of floats
            Endpoints of integration interval in which to calculate KLD
        dx: float
            resolution of integration grid

        Returns
        -------
        KL: float
            Value of Kullback-Leibler divergence from approximation to truth if truth is available; otherwise nothing.

        Notes
        -----
        Example:
            d = p.kld(limits=(-1., 1.), dx=1./100))
        """

        if self.truth is None:
            print('Truth not available for comparison.')
            return
        else:
            KL = qp.utils.calculate_kl_divergence(self, self, limits=limits, dx=dx)
            return(KL)

    def rms(self, limits=(0., 1.), dx=0.01):
        """
        Calculates root mean square difference between quantile approximation and truth.

        Parameters
        ----------
        limits: tuple of floats
            Endpoints of integration interval in which to calculate KLD
        dx: float
            resolution of integration grid

        Returns
        -------
        RMS: float
            Value of root mean square difference between approximation of truth if truth is available; otherwise nothing.

        Notes
        -----
        Example:
            d = p.rms(limits=(-1., 1.), dx=1./100))
        """

        if self.truth is None:
            print('Truth not available for comparison.')
            return
        else:
            RMS = qp.utils.calculate_rms(self, self, limits=limits, dx=dx)
            return(RMS)
