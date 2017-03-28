import numpy as np
import scipy.stats as sps
import scipy.interpolate as spi
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn import mixture

import qp

class PDF(object):

    def __init__(self, truth=None, quantiles=None, histogram=None,
                 gridded=None, samples=None, scheme='linear',
                 vb=True):
        """
        An object representing a probability density function in
        various ways.

        Parameters
        ----------
        truth: scipy.stats.rv_continuous object or qp.composite object, optional
            Continuous, parametric form of the PDF
        quantiles: tuple of ndarrays, optional
            Pair of arrays of lengths (nquants, nquants) containing CDF
            values and quantiles
        histogram: tuple of ndarrays, optional
            Pair of arrays of lengths (nbins+1, nbins) containing
            endpoints of bins and values in bins
        gridded: tuple of ndarrays, optional
            Pair of arrays of lengths (npoints, npoints) containing
            points at which function is evaluated and function values
            at those points
        samples: ndarray, optional
            Array of length nsamples containing sampled values
        scheme: string, optional
            name of interpolation scheme to use.
        vb: boolean
            report on progress to stdout?
        """
        self.truth = truth
        self.quantiles = quantiles
        self.histogram = histogram
        self.samples = samples
        self.gridded = gridded
        self.mix_mod = None

        self.scheme = scheme

        if vb and self.truth is None and self.quantiles is None and self.histogram is None and self.gridded is None and self.samples is None:
            print 'Warning: initializing a PDF object without inputs'
            return

        # Record how the PDF object was initialized:
        if self.truth is not None:
            self.initialized = self.truth
            self.first = 'truth'
        elif self.quantiles is not None:
            self.initialized = self.quantiles
            self.first = 'quantiles'
        elif self.histogram is not None:
            self.initialized = self.histogram
            self.first = 'histogram'
        elif self.gridded is not None:
            self.initialized = self.gridded
            self.first = 'gridded'
        elif self.samples is not None:
            self.initialized = self.samples
            self.first = 'samples'

        # The most recent parametrization used is, at this point, the
        # first one:
        self.last = self.first

        # We'll make an interpolator if and when we need it:
        self.interpolator = None

        return

    def evaluate(self, loc, vb=True, using=None):
        """
        Evaluates the PDF (either the true version or the first
        approximation of it if no parametrization is specified)
        at the given location(s).

        Parameters
        ----------
        loc: float or ndarray
            location(s) at which to evaluate the pdf
        vb: boolean
            report on progress to stdout?
        using: string
            which parametrization to evaluate, defaults to initialization

        Returns
        -------
        val: float or ndarray
            the value of the PDF (ot its approximation) at the requested location(s)
        """
        if using is None:
            using = self.first

        if using == 'truth':
            if self.truth is not None:
                if vb: print 'Evaluating the true distribution.'
                val = self.truth.pdf(loc)
                self.evaluated = (loc, val)
            else:
                raise ValueError('true PDF is not set, use an approximation instead (the most recent one was '+self.last+')')
        elif using == 'mix_mod':
            if self.mix_mod is None:
                self.mix_mod = self.mix_mod_fit()
            if vb: print 'Evaluating the fitted mixture model distribution.'
            val = self.mix_mod.pdf(loc)
            self.evaluated = (loc, val)
        else:
            if vb: print 'Evaluating a `'+self.scheme+'` interpolation of the '+using+' parametrization.'
            evaluated = self.approximate(loc, using=using, vb=vb)
            val = evaluated[1]

        return(loc, val)

    def integrate(self, limits):
        """
        Computes the integral under the PDF between the given limits.

        Parameters
        ----------
        limits: float, tuple
            limits of integration

        Returns
        -------
        integral: float
            value of the integral

        Notes
        -----
        This method is not yet operational, and returns `None`.
        """
        return None

    def quantize(self, quants=None, percent=10., number=None, infty=100., vb=True):
        """
        Computes an array of evenly-spaced quantiles from the truth.

        Parameters
        ----------
        quants: ndarray, float, optional
            array of quantile locations as decimals
        percent: float, optional
            the separation of the requested quantiles, in percent
        number: int, optional
            the number of quantiles to compute.
        infty: float, optional
            approximate value at which CDF=1.
        vb: boolean
            report on progress to stdout?

        Returns
        -------
        self.quantiles: ndarray, float
            the quantile points.

        Notes
        -----
        Quantiles of a PDF could be a useful approximate way to store it.
        This method computes the quantiles from a truth distribution
        (other representations forthcoming)
        and stores them in the `self.quantiles` attribute.
        Uses the `.ppf` method of the `rvs_continuous` distribution
        object stored in `self.truth`. This calculates the inverse CDF.
        See `the Scipy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.ppf.html#scipy.stats.rv_continuous.ppf>`_ for details.
        """
        if quants is not None:
            quantpoints = quants
        else:
            if number is not None:
                # Compute the spacing of the quantiles:
                quantum = 1.0 / float(number+1)
            else:
                quantum = percent/100.0
                # Over-write the number of quantiles:
                number = np.ceil(100.0 / percent) - 1
                assert number > 0
            quantpoints = np.linspace(0.0+quantum, 1.0-quantum, number)

        if vb:
            print("Calculating "+str(len(quantpoints))+" quantiles: "+str(quantpoints))
        if self.truth is not None:
            quantiles = self.truth.ppf(quantpoints)
        else:
            print('New quantiles can only be computed from a truth distribution in this version.')
            return

        if vb:
            print("Resulting "+str(len(quantiles))+" quantiles: "+str(quantiles))
            integrals = self.truth.cdf(quantiles)
            print("Checking integrals: "+str(integrals))
        self.quantiles = (quantpoints, quantiles)
        print(np.shape(quantpoints), np.shape(quantiles))
        self.last = 'quantiles'
        return self.quantiles

    def histogramize(self, binends=None, number=10, binrange=[0., 1.], vb=True):
        """
        Computes integrated histogram bin values from the truth via the CDF.

        Parameters
        ----------
        binends: ndarray, float, optional
            Array of N+1 endpoints of N bins
        number: int, optional
            Number of bins if no binends provided
        range: tuple, float, optional
            Pair of values of endpoints of total bin range
        vb: boolean
            Report on progress to stdout?

        Returns
        -------
        self.histogram: tuple of ndarrays of floats
            Pair of arrays of lengths (number+1, number) containing endpoints
            of bins and values in bins

        Comments
        --------
        A histogram representation of a PDF is a popular approximate way
        to store it. This method computes some histogram bin heights
        from a truth distribution (other representations forthcoming)
        and stores them in the `self.histogram` attribute.
        Uses the `.cdf` method of the `rvs_continuous` distribution
        object stored in `self.truth`. This calculates the CDF.
        See `the Scipy docs <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.cdf.html#scipy.stats.rv_continuous.cdf>`_ for details.
        """
        if binends is None:
            step = float(binrange[1]-binrange[0])/number
            binends = np.arange(binrange[0], binrange[1]+step, step)

        number = len(binends)-1
        histogram = np.zeros(number)
        if vb: print 'Calculating histogram: ', binends
        if self.truth is not None:
            cdf = self.truth.cdf(binends)
            for b in range(number):
                histogram[b] = (cdf[b+1]-cdf[b])/(binends[b+1]-binends[b])
        else:
            print 'New histograms can only be computed from a truth distribution in this version.'
            return

        if vb: print 'Result: ', histogram
        self.histogram = (binends, histogram)
        self.last = 'histogram'
        return self.histogram

    def mix_mod_fit(self, n_components=5, using=None):
        """
        Fits the parameters of a given functional form to an approximation

        Parameters
        ----------
        n_components: int, optional
            number of components to consider
        using: string, optional
            which existing approximation to use, defaults to first approximation

        Returns
        -------
        self.mix_mod: qp.composite object
            the qp.composite object approximating the PDF

        Notes
        -----
        Currently only supports mixture of Gaussians
        """
        if self.samples is None:
            self.samples = self.sample(using=using)

        estimator = skl.mixture.GaussianMixture(n_components=n_components)
        estimator.fit(self.samples.reshape(-1, 1))

        weights = estimator.weights_
        means = estimator.means_
        variances = estimator.covariances_

        components = []
        for i in range(n_components):
            mix_mod_dict = {}
            function = sps.norm(loc = means[i], scale = np.sqrt(variances[i][0][0]))
            coefficient = weights[i]
            mix_mod_dict['function'] = function
            mix_mod_dict['coefficient'] = coefficient
            components.append(mix_mod_dict)

        self.mix_mod = qp.composite(components)
        return self.mix_mod

    def sample(self, N=100, infty=100., using=None, vb=True):
        """
        Samples the pdf in given representation

        Parameters
        ----------
        N: int, optional
            number of samples to produce
        infty: float, optional
            approximate value at which CDF=1.
        using: string, optional
            Parametrization on which to interpolate, defaults to initialization
        vb: boolean
            report on progress to stdout?

        Returns
        -------
        samples: ndarray
            array of sampled values
        """
        if using is None:
            using = self.first

        if vb: print 'Sampling from '+using+' parametrization.'

        if using == 'truth':
            samples = self.truth.rvs(size=N)
        elif using == 'mix_mod':
            samples = self.mix_mod.rvs(size=N)

        else:
            if using == 'quantiles':
                # First find the quantiles if none exist:
                if self.quantiles is None:
                    self.quantiles = self.quantize()

                endpoints = np.append(np.array([-1.*infty]), self.quantiles[1])
                endpoints = np.append(endpoints,np.array([infty]))
                weights = qp.utils.evaluate_quantiles(self.quantiles, infty=infty)[1]# self.evaluate((endpoints[1:]+endpoints[:-1])/2.)

            if using == 'histogram':
                # First find the histogram if none exists:
                if self.histogram is None:
                    self.histogram = self.histogramize()

                endpoints = self.histogram[0]
                weights = self.histogram[1]

            ncats = len(weights)
            cats = range(ncats)
            sampbins = [0]*ncats
            for item in range(N):
                sampbins[qp.utils.choice(cats, weights)] += 1
            samples = []*N
            for c in cats:
                for n in range(sampbins[c]):
                    samples.append(np.random.uniform(low=endpoints[c], high=endpoints[c+1]))

        if vb: print 'Sampled values: ', samples
        self.samples = samples
        self.last = 'samples'
        return self.samples

    def interpolate(self, using=None, vb=True):
        """
        Constructs an `interpolator` function based on the parametrization.

        Parameters
        ----------
        using: string, optional
            parametrization on which to interpolate, defaults to initialization
        vb: boolean
            report on progress to stdout?

        Returns
        -------
        self.interpolator
            an interpolator object

        Notes
        -----
        The `self.interpolator` object is a function that is used by the
        `approximate` method. It employs
        [`scipy.interpolate.interp1d`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html)
        to carry out the interpolation, using the internal
        `self.scheme` attribute to choose the interpolation scheme.
        """
        if using is None:
            using = self.first
        # if vb: print('Interpolating the `'+using+'` parametrization')

        if using == 'truth' or using == 'mix_mod':
            print 'A functional form needs no interpolation.  Try converting to an approximate parametrization first.'
            return

        if using == 'quantiles':
            # First find the quantiles if none exist:
            if self.quantiles is None:
                self.quantiles = self.quantize()

            (x, y) = qp.utils.evaluate_quantiles(self.quantiles)

        if using == 'histogram':
            # First find the histogram if none exists:
            if self.histogram is None:
                self.histogram = self.histogramize()

            (x, y) = qp.utils.evaluate_histogram(self.histogram)

        if using == 'gridded':
            if self.gridded is None:
                print 'Interpolation from a gridded parametrization requires a previous gridded parametrization.'
                return
            (x, y) = self.gridded

        if using == 'samples':
            # First sample if not already done:
            if self.samples is None:
                self.samples = self.sample()

            (x, y) = qp.evaluate_samples(self.samples)

        if vb:
            print 'Creating a `'+self.scheme+'` interpolator for the '+using+' parametrization.'

        self.interpolator = spi.interp1d(x, y, kind=self.scheme,
                                         fill_value="extrapolate")

        return self.interpolator

    def approximate(self, points, using=None, scheme=None, vb=True):
        """
        Interpolates the parametrization to get an approximation to the density.

        Parameters
        ----------
        number: int
            the number of points over which to interpolate, bounded by
            the quantile value endpoints
        points: ndarray
            the value(s) at which to evaluate the interpolated function
        using: string, optional
            approximation parametrization, currently either 'quantiles'
            or 'histogram'
        scheme: string, optional
            interpolation scheme, from the [`scipy.interpolate.interp1d`
            options](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html).
            If passed as `None`, the internal `self.scheme` attribute
            is used - this defaults to `linear` in the constructor.
            Otherwise, this attribute is reset to the one chosen.
        vb: boolean
            report on progress to stdout?

        Returns
        -------
        points: ndarray, float
            the input grid upon which to interpolate
        interpolated: ndarray, float
            the interpolated points.

        Notes
        -----
        Extrapolation is via the `scheme` while values are positive;
        otherwise, extrapolation returns 0.

        Example:
            x, y = p.approximate(np.linspace(-1., 1., 100))
        """

        # First, reset the interpolation scheme if one is passed
        # explicitly:
        if scheme is not None:
            self.scheme = scheme

        # Now make the interpolation, using the current scheme:
        self.interpolator = self.interpolate(using=using, vb=vb)
        interpolated = self.interpolator(points)
        interpolated[interpolated<0.] = 0.

        return (points, interpolated)

    def plot(self, vb=True):
        """
        Plots the PDF, in various ways.

        Parameters
        ----------
        vb: boolean
            report on progress to stdout?

        Notes
        -----
        What this method plots depends on what information about the PDF
        is stored in it: the more properties the PDF has,
        the more exciting the plot!
        """
        extrema = [0., 0.]

        if self.truth is not None:
            min_x = self.truth.ppf(np.array([0.001]))
            max_x = self.truth.ppf(np.array([0.999]))
            x = np.linspace(min_x, max_x, 100)
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            y = self.truth.pdf(x)
            plt.plot(x, y, color='k', linestyle='-', lw=1.0, alpha=1.0, label='True PDF')
            if vb:
                print 'Plotted truth.'

        if self.mix_mod is not None:
            min_x = self.mix_mod.ppf(np.array([0.001]))
            max_x = self.mix_mod.ppf(np.array([0.999]))
            x = np.linspace(min_x, max_x, 100)
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            y = self.mix_mod.pdf(x)
            plt.plot(x, y, color='k', linestyle=':', lw=1.0, alpha=0.5, label='Mixture Model PDF')
            if vb:
                print 'Plotted mixture model.'

        if self.quantiles is not None:
            min_x = self.quantiles[1][0]
            max_x = self.quantiles[1][-1]
            x = np.linspace(min_x, max_x, 100)
            plt.vlines(self.quantiles[1],
                       np.zeros(len(self.quantiles[1])),
                       self.evaluate(self.quantiles[1],
                                     using='quantiles', vb=False)[1],
                       color='b', linestyle=':', lw=1.0, alpha=1.0,
                       label='Quantiles')
            (grid, qinterpolated) = self.approximate(x, vb=vb,
                                                     using='quantiles')
            plt.plot(grid, qinterpolated, color='b', lw=2.0, alpha=1.0,
                     linestyle=(0,(5,10)),
                     label='Quantile Interpolated PDF')
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            if vb:
                print 'Plotted quantiles.'

        if self.histogram is not None:
            min_x = self.histogram[0][0]
            max_x = self.histogram[0][-1]
            x = np.linspace(min_x, max_x, 100)
            plt.hlines(self.histogram[1], self.histogram[0][:-1],
                       self.histogram[0][1:], color='r', linestyle=':',
                       lw=1.0, alpha=1., label='Histogram')
            (grid, hinterpolated) = self.approximate(x, vb=vb,
                                                     using='histogram')
            plt.plot(grid, hinterpolated, color='r', lw=2.0, alpha=1.0,
                     linestyle=(5,(5,10)),
                     label='Histogram Interpolated PDF')
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            if vb:
                print 'Plotted histogram.'

        if self.gridded is not None:
            min_x = self.gridded[0][0]
            max_x = self.gridded[0][-1]
            (x, y) = self.gridded
            plt.plot(x, y, color='k', lw=2.0, alpha=0.5,
                     linestyle='--', label='gridded PDF')
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            if vb:
                print 'Plotted evaluation.'

        if self.samples is not None:
            min_x = min(self.samples)
            max_x = max(self.samples)
            x = np.linspace(min_x, max_x, 100)
            plt.plot(self.samples, np.zeros(np.shape(self.samples)),
                     'g+', ms=20, label='Samples')
            (grid, sinterpolated) = self.approximate(x, vb=vb,
                                                     using='samples')
            plt.plot(grid, sinterpolated, color='g', lw=2.0, alpha=1.0,
                     linestyle=(10,(5,10)),
                     label='Samples Interpolated PDF')
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            if vb:
                print('Plotted samples')

        plt.xlim(extrema[0], extrema[-1])
        plt.legend(fontsize='small')
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
            endpoints of integration interval in which to calculate KLD
        dx: float
            resolution of integration grid

        Returns
        -------
        KL: float
            value of Kullback-Leibler divergence from approximation to truth
            if truth is available; otherwise nothing.

        Notes
        -----
        Example::
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
        Calculates root mean square difference between quantile approximation
        and truth.

        Parameters
        ----------
        limits: tuple of floats
            endpoints of integration interval in which to calculate KLD
        dx: float
            resolution of integration grid

        Returns
        -------
        RMS: float
            value of root mean square difference between approximation of truth
            if truth is available; otherwise nothing.

        Notes
        -----
        Example::
            d = p.rms(limits=(-1., 1.), dx=1./100))
        """

        if self.truth is None:
            print('Truth not available for comparison.')
            return
        else:
            RMS = qp.utils.calculate_rms(self, self, limits=limits, dx=dx)
            return(RMS)
