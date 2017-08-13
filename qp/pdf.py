import numpy as np
import bisect
import scipy.stats as sps
import scipy.interpolate as spi
import scipy.optimize as spo
import matplotlib.pyplot as plt
import sklearn as skl
from sklearn import mixture

import qp
from qp.utils import infty as default_infty
from qp.utils import epsilon as default_eps
from qp.utils import lims as default_lims

class PDF(object):

    def __init__(self, truth=None, quantiles=None, histogram=None,
                 gridded=None, samples=None, limits=default_lims, scheme='linear',
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
        limits: tuple, float, optional
            limits past which PDF is considered to be 0.
        scheme: string, optional
            name of interpolation scheme to use.
        vb: boolean
            report on progress to stdout?

        Notes
        -----
        TO DO: enable truth to be any parametrization
        TO DO: change dx --> dz (or delta)
        TO DO: consider changing quantiles input to just be the z-values since interpolation gives the p(z) values anyway
        """
        self.truth = truth
        self.quantiles = quantiles
        self.histogram = qp.utils.normalize_histogram(histogram, vb=vb)
        self.samples = samples
        self.gridded = qp.utils.normalize_integral(qp.utils.normalize_gridded(gridded, vb=vb))
        self.mix_mod = None
        self.limits = limits

        self.scheme = scheme

        if vb and self.truth is None and self.quantiles is None and self.histogram is None and self.gridded is None and self.samples is None:
            print 'Warning: initializing a PDF object without inputs'
            return

        # Record how the PDF object was initialized:
        if self.truth is not None:
            self.initialized = self.truth
            self.first = 'truth'
        elif self.gridded is not None:
            self.initialized = self.gridded
            self.first = 'gridded'
            self.limits = (min(self.limits[0], np.min(self.gridded[0])), max(self.limits[-1], np.max(self.gridded[0])))
        elif self.samples is not None:
            self.initialized = self.samples
            self.first = 'samples'
            self.limits = (min(self.limits[0], np.min(self.samples)), max(self.limits[-1], np.max(self.samples)))
        elif self.histogram is not None:
            self.initialized = self.histogram
            self.first = 'histogram'
            self.limits = (min(self.limits[0], np.min(self.histogram[0])), max(self.limits[-1], np.max(self.histogram[0])))
        elif self.quantiles is not None:
            self.initialized = self.quantiles
            self.first = 'quantiles'
            self.limits = (min(self.limits[0], np.min(self.quantiles[-1])), max(self.limits[-1], np.max(self.quantiles[-1])))

        # The most recent parametrization used is, at this point, the
        # first one:
        self.last = self.first

        # We'll make an interpolator if and when we need it:
        self.interpolator = None

        return

    def evaluate(self, loc, using=None, norm=False, vb=True):
        """
        Evaluates the PDF (either the true version or the first
        approximation of it if no parametrization is specified)
        at the given location(s).

        Parameters
        ----------
        loc: float or ndarray
            location(s) at which to evaluate the pdf
        using: string
            which parametrization to evaluate, defaults to initialization
        norm: boolean, optional
            True to normalize the evaluation, False if expected probability outside loc
        vb: boolean
            report on progress to stdout?

        Returns
        -------
        (loc, val): tuple, float or ndarray
            the input locations and the value of the PDF (or its approximation) at the requested location(s)
        """
        if using is None:
            using = self.last

        if using == 'truth':
            if self.truth is not None:
                if vb: print 'Evaluating the true distribution.'
                val = self.truth.pdf(loc)
                self.evaluated = (loc, val)
            else:
                raise ValueError('true PDF is not set, use an approximation instead (the most recent one was '+self.last+')')
        elif using == 'mix_mod':
            if self.mix_mod is None:
                self.mix_mod = self.mix_mod_fit(vb=vb)
            if vb: print 'Evaluating the fitted mixture model distribution.'
            val = self.mix_mod.pdf(loc)
            self.evaluated = (loc, val)
        else:
            # if vb: print 'Evaluating a `'+self.scheme+'` interpolation of the '+using+' parametrization.'
            evaluated = self.approximate(loc, using=using, vb=vb)
            val = evaluated[1]

        gridded = (loc, val)
        if norm:
            gridded = qp.utils.normalize_integral(gridded)

        return gridded

    def integrate(self, limits=None, dx=0.001, using=None, vb=True):
        """
        Computes the integral under the PDF between the given limits.

        Parameters
        ----------
        limits: tuple, float, optional
            limits of integration
        dx: float, optional
            granularity of integral
        using: string, optional
            parametrization over which to approximate the integral
        vb: Boolean
            print progress to stdout?

        Returns
        -------
        integral: float
            value of the integral
        """
        if limits is None:
            limits = self.limits
            if vb:
                print('Integrating over '+str(limits)+' because no limits provided')
        lim_range = limits[-1] - limits[0]
        fine_grid = np.arange(limits[0], limits[-1] + dx, dx)

        evaluated = self.evaluate(fine_grid, vb=vb, using=using)
        integral = np.sum(evaluated[1]) * dx

        return integral

    def quantize(self, quants=None, N=9, limits=None, vb=True):
        """
        Computes an array of evenly-spaced quantiles from the truth.

        Parameters
        ----------
        quants: ndarray, float, optional
            array of quantile locations as decimals
        N: int, optional
            number of regular quantiles to compute
        limits: tuple, float, optional
            approximate values at which CDF=0. and CDF=1.
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
            N = len(quantpoints)
        elif N is not None:
            quantum = 1. / float(N+1)
            quantpoints = np.linspace(0.+quantum, 1.-quantum, N)

        if vb:
            print("Calculating "+str(len(quantpoints))+" quantiles: "+str(quantpoints))

        if self.truth is not None:
            if isinstance(self.truth, qp.composite):
                if limits is None:
                    limits = self.limits
                extrapoints = np.concatenate((np.array([0.]), quantpoints, np.array([1.])))
                min_delta = np.min(extrapoints[1:] - extrapoints[:-1])
                n_grid = N
                icdf = extrapoints
                while np.max(icdf[1:] - icdf[:-1]) >= min_delta:
                    grid = np.linspace(limits[0], limits[-1], n_grid)
                    icdf = self.truth.cdf(grid)#np.array([self.integrate(limits=(limits[0], grid[i]), dx=spacing/10., using='truth', vb=vb) for i in range(len(grid))])
                    n_grid *= 10
                locs = np.array([bisect.bisect_right(icdf[:-1], quantpoints[n]) for n in range(N)])

                quantiles = self.truth.ppf(quantpoints, ivals=grid[locs])
            else:
                quantiles = self.truth.ppf(quantpoints)
        else:
            print('New quantiles can only be computed from a truth distribution in this version.')
            return

        if vb:
            print("Resulting "+str(len(quantiles))+" quantiles: "+str(quantiles))
        # integrals = self.truth.cdf(quantiles)
        # assert np.isclose(integrals, quantpoints)
        self.quantiles = (quantpoints, quantiles)
        self.limits = (min(limits[0], np.min(quantiles)), max(limits[-1], np.max(quantiles)))
        self.last = 'quantiles'
        return self.quantiles

    def histogramize(self, binends=None, N=10, binrange=(-1.*default_infty, default_infty), vb=True):
        """
        Computes integrated histogram bin values from the truth via the CDF.

        Parameters
        ----------
        binends: ndarray, float, optional
            array of N+1 endpoints of N bins
        N: int, optional
            number of regular bins if no binends provided
        binrange: tuple, float, optional
            pair of values of endpoints of total bin range
        vb: boolean
            report on progress to stdout?

        Returns
        -------
        self.histogram: tuple of ndarrays of floats
            Pair of arrays of lengths (N+1, N) containing endpoints
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
        if binrange is None:
            if self.gridded is not None:
                binrange = [min(self.gridded[0]), max(self.gridded[0])]
            elif self.samples is not None:
                binrange = [min(self.samples), max(self.samples)]
            elif self.quantiles is not None:
                binrange = [min(self.quantiles[1]), max(self.quantiles[1])]
            elif self.histogram is not None:
                return self.histogram
            else:
                binrange = [0., 1.]

        if binends is None:
            step = float(binrange[1]-binrange[0])/N
            binends = np.arange(binrange[0], binrange[1]+step, step)

        N = len(binends)-1
        histogram = np.zeros(N)
        if vb: print 'Calculating histogram: ', binends
        if self.truth is not None:
            cdf = self.truth.cdf(binends)
            heights = cdf[1:] - cdf[:-1]
            histogram = qp.utils.normalize_histogram((binends, heights), vb=vb)
            # for b in range(N):
            #     histogram[b] = (cdf[b+1]-cdf[b])/(binends[b+1]-binends[b])
        else:
            print 'New histograms can only be computed from a truth distribution in this version.'
            return

        if vb: print 'Result: ', histogram
        self.histogram = histogram
        self.last = 'histogram'
        return self.histogram

    def mix_mod_fit(self, n_components=5, using=None, vb=True):
        """
        Fits the parameters of a given functional form to an approximation

        Parameters
        ----------
        n_components: int, optional
            number of components to consider
        using: string, optional
            which existing approximation to use, defaults to first approximation
        vb: boolean
            Report progress on stdout?

        Returns
        -------
        self.mix_mod: qp.composite object
            the qp.composite object approximating the PDF

        Notes
        -----
        Currently only supports mixture of Gaussians
        TO DO: change syntax n_components --> N
        """
        comp_range = range(n_components)
        if using == None:
            using = self.first

        if using == 'gridded':
            if self.gridded is not None:
                (x, y) = self.gridded
                ival_weights = np.ones(n_components) / n_components
                ival_means = min(x) + (max(x) - min(x)) * np.arange(n_components) / n_components
                ival_stdevs = np.sqrt((max(x) - min(x)) * np.ones(n_components) / n_components)
                ivals = np.array([ival_weights, ival_means, ival_stdevs]).T.flatten()
                def gmm(x, *args):
                    y = 0.
                    args = np.array(args).reshape((n_components, 3))
                    for c in comp_range:
                        # index = c * n_components
                        y += args[c][0] *  sps.norm(loc = args[c][1], scale = args[c][2]).pdf(x)
                    return y
                low_bounds = np.array([np.zeros(n_components), min(x) * np.ones(n_components), np.ones(n_components) * (max(x) - min(x)) / len(x)]).T.flatten()
                high_bounds = np.array([np.ones(n_components), max(x) * np.ones(n_components), np.ones(n_components) * (max(x) - min(x))]).T.flatten()
                popt, pcov = spo.curve_fit(gmm, self.gridded[0], self.gridded[1], ivals, bounds = (low_bounds, high_bounds))
                popt = popt.reshape((n_components, 3)).T
                weights = popt[0]
                means = popt[1]
                stdevs = popt[2]
            else:
                print('No gridded parametrization available.  Try using a different format.')
                return
        else:
            if self.samples is None:
                self.samples = self.sample(using=using, vb=vb)

            estimator = skl.mixture.GaussianMixture(n_components=n_components)
            estimator.fit(self.samples.reshape(-1, 1))

            weights = estimator.weights_
            means = estimator.means_[:, 0]
            stdevs = np.sqrt(estimator.covariances_[:, 0, 0])

        if vb:
            print(weights, means, stdevs)

        components = []
        for i in comp_range:
            mix_mod_dict = {}
            function = sps.norm(loc = means[i], scale = stdevs[i])
            coefficient = weights[i]
            mix_mod_dict['function'] = function
            mix_mod_dict['coefficient'] = coefficient
            components.append(mix_mod_dict)

        if vb:
            statement = ''
            for c in comp_range:
                statement += str(weights[c])+r'$\cdot\mathcal{N}($'+str(means[c])+r','+str(stdevs[c])+r')\n'
            print(statement)
        self.mix_mod = qp.composite(components)
        return self.mix_mod

    def sample(self, N=1000, infty=default_infty, using=None, vb=True):
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

        Notes
        -----
        TO DO: have quantiles use linear interpolator to get inverse CDF, then sample uniform in "bins"
        TO DO: change infty to upper and lower bounds to use for quantiles
        TO DO: check for existence of parametrization before using it
        """
        if using is None:
            using = self.first

        if vb: print 'Sampling from '+using+' parametrization.'

        if using == 'truth':
            samples = self.truth.rvs(size=N)
        elif using == 'mix_mod':
            samples = self.mix_mod.rvs(size=N)

        elif using == 'gridded':
            interpolator = self.interpolate(using = 'gridded', vb=vb)
            # (xlims, ylims) = self.evaluate(self.limits, using='gridded', vb=vb)
            (xmin, xmax) = (min(self.gridded[0]), max(self.gridded[0]))
            (ymin, ymax) = (min(self.gridded[1]), max(self.gridded[1]))
            (xran, yran) = (xmax - xmin, ymax - ymin)
            samples = []
            while len(samples) < N:
                (x, y) = (xmin + xran * np.random.uniform(), ymin + yran * np.random.uniform())
                if y < interpolator(x):
                    samples.append(x)

        else:
            if using == 'quantiles':
                # First find the quantiles if none exist:
                if self.quantiles is None:
                    self.quantiles = self.quantize(vb=vb)

                (x, y) = qp.utils.evaluate_quantiles(self.quantiles)
                (endpoints, weights) = qp.utils.normalize_quantiles(self.quantiles[0], (x, y))
                # endpoints = np.insert(self.quantiles[1], [0, -1], self.limits)
                # weights = qp.utils.evaluate_quantiles(self.quantiles)[1]# self.evaluate((endpoints[1:]+endpoints[:-1])/2.)
                # interpolator = self.interpolate(using='quantiles', vb=False)

            if using == 'histogram':
                # First find the histogram if none exists:
                if self.histogram is None:
                    self.histogram = self.histogramize(vb=vb)

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
        self.samples = np.array(samples)
        self.limits = (min(self.limits[0], np.min(self.samples)), max(self.limits[-1], np.max(self.samples)))
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
        TO DO: There's got to be a better to do quantile interpolation!  Maybe use inverse CDF?
        """
        if using is None:
            using = self.last

        if using == 'truth' or using == 'mix_mod':
            print 'A functional form needs no interpolation.  Try converting to an approximate parametrization first.'
            return

        if using == 'quantiles':
            # First find the quantiles if none exist:
            if self.quantiles is None:
                self.quantiles = self.quantize(vb=vb)

            (x, y) = qp.utils.evaluate_quantiles(self.quantiles)
            (x, y) = qp.utils.normalize_quantiles(self.quantiles[0], (x, y))
            self.interpolator = spi.interp1d(x, y, kind=self.scheme, bounds_error=False, fill_value=default_eps)
            return self.interpolator

        if using == 'histogram':
            # First find the histogram if none exists:
            if self.histogram is None:
                self.histogram = self.histogramize(vb=vb)

            extra_y = np.insert(self.histogram[1], [0, -1], (default_eps, default_eps))

            def histogram_interpolator(xf):
                nx = len(xf)
                yf = np.ones(nx) * default_eps
                for n in range(nx):
                    i = bisect.bisect_left(self.histogram[0], xf[n])
                    yf[n] = extra_y[i]
                return(yf)

            #(x, y) = qp.utils.evaluate_histogram(self.histogram)

            self.interpolator = histogram_interpolator#qp.utils.evaluate_histogram()

            if vb:
                print 'Created a piecewise constant interpolator for the '+using+' parametrization.'

            return self.interpolator

        if using == 'samples':
            # First sample if not already done:
            if self.samples is None:
                self.samples = self.sample(vb=vb)

            # Well that's weird!  Samples does much better with KDE than linear interpolation.  I guess that shouldn't be surprising.
            def samples_interpolator(xf):
                kde = sps.gaussian_kde(self.samples)# , bw_method=bandwidth)
                yf = kde(xf)
                return (yf)
            self.interpolator = samples_interpolator
            if vb:
                print 'Created a KDE interpolator for the '+using+' parametrization.'

            # (x, y) = qp.evaluate_samples(self.samples)
            # self.interpolator = spi.interp1d(x, y, kind=self.scheme, bounds_error=False, fill_value=default_eps)
            # if vb: print('interpolator support between '+str(min(x))+' and '+str(max(x))+' with extrapolation of '+str(default_eps))

            return self.interpolator

        if using == 'gridded':
            if self.gridded is None:
                print 'Interpolation from a gridded parametrization requires a previous gridded parametrization.'
                return
            (x, y) = self.gridded

        self.interpolator = spi.interp1d(x, y, kind=self.scheme, fill_value="extrapolate")

        if vb:
            print 'Created a `'+self.scheme+'` interpolator for the '+using+' parametrization.'

        return self.interpolator

    def approximate(self, points, using=None, scheme=None, vb=True):
        """
        Interpolates the parametrization to get an approximation to the density.

        Parameters
        ----------
        points: ndarray
            the value(s) at which to evaluate the interpolated function
        using: string, optional
            approximation parametrization
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
        if vb: print('interpolating between '+str(min(points))+' and '+str(max(points))+' using '+using)
        interpolated = self.interpolator(points)
        interpolated = qp.utils.normalize_gridded((points, interpolated), vb=vb)
        # interpolated[interpolated<0.] = 0.

        return interpolated#(points, interpolated)

    def plot(self, limits=None, loc='plot.png', vb=True):
        """
        Plots the PDF, in various ways.

        Parameters
        ----------
        limits: tuple, float, optional
            limits over which plot should be made
        loc: string, optional
            filepath/name for saving the plot, optional
        vb: boolean, optional
            report on progress to stdout?

        Notes
        -----
        What this method plots depends on what information about the PDF
        is stored in it: the more properties the PDF has,
        the more exciting the plot!
        """
        if limits is None:
            limits = self.limits
        extrema = limits

        colors = {}
        colors['truth'] = 'k'
        colors['mix_mod'] = 'k'
        colors['gridded'] = 'k'
        colors['quantiles'] = 'b'#'blueviolet'
        colors['histogram'] = 'r'#'darkorange'
        colors['samples'] = 'g'#'forestgreen'

        styles = {}
        styles['truth'] = '-'
        styles['mix_mod'] = ':'
        styles['gridded'] = '--'
        styles['quantiles'] = (0,(5,10))
        styles['histogram'] = (0,(3,6))
        styles['samples'] = (0,(1,2))

        if self.truth is not None:
            min_x = self.truth.ppf(np.array([0.001]))
            max_x = self.truth.ppf(np.array([0.999]))
            x = np.linspace(min_x, max_x, 100)
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            y = self.truth.pdf(x)
            plt.plot(x, y, color=colors['truth'], linestyle=styles['truth'], lw=1.0, alpha=1.0, label='True PDF')
            if vb:
                print 'Plotted truth.'

        if self.mix_mod is not None:
            min_x = self.mix_mod.ppf(np.array([0.001]))
            max_x = self.mix_mod.ppf(np.array([0.999]))
            x = np.linspace(min_x, max_x, 100)
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            y = self.mix_mod.pdf(x)
            plt.plot(x, y, color=colors['mix_mod'], linestyle=styles['mix_mod'], lw=2.0, alpha=1.0, label='Mixture Model PDF')
            if vb:
                print 'Plotted mixture model.'

        if self.quantiles is not None:
            (z, p) = self.evaluate(self.quantiles[1], using='quantiles', vb=vb)
            (x, y) = qp.utils.normalize_quantiles(self.quantiles[0], (z, p))
            min_x = min(min(x), extrema[0])
            max_x = max(max(x), extrema[-1])
            x = np.linspace(min_x, max_x, 100)
            (grid, qinterpolated) = self.approximate(x, vb=vb, using='quantiles')
            plt.vlines(z, np.zeros(len(self.quantiles[1])), p, color=colors['quantiles'], linestyle=':', lw=1.0, alpha=1.0, label='Quantiles')
            plt.plot(grid, qinterpolated, color=colors['quantiles'], lw=2.0, alpha=1.0, linestyle=styles['quantiles'], label='Quantile Interpolated PDF')
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            if vb:
                print 'Plotted quantiles.'

        if self.histogram is not None:
            min_x = self.histogram[0][0]
            max_x = self.histogram[0][-1]
            x = np.linspace(min_x, max_x, 100)
            plt.hlines(self.histogram[1], self.histogram[0][:-1],
                       self.histogram[0][1:], color=colors['histogram'], linestyle=':', lw=1.0, alpha=1., label='Histogram')
            (grid, hinterpolated) = self.approximate(x, vb=vb,
                                                     using='histogram')
            plt.plot(grid, hinterpolated, color=colors['histogram'], lw=2.0, alpha=1.0,
                     linestyle=styles['histogram'],
                     label='Histogram Interpolated PDF')
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            if vb:
                print 'Plotted histogram.'

        if self.gridded is not None:
            min_x = min(self.gridded[0])
            max_x = max(self.gridded[0])
            (x, y) = self.gridded
            plt.plot(x, y, color=colors['gridded'], lw=2.0, alpha=0.5,
                     linestyle=styles['gridded'], label='Gridded PDF')
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            if vb:
                print 'Plotted gridded.'

        if self.samples is not None:
            min_x = min(self.samples)
            max_x = max(self.samples)
            x = np.linspace(min_x, max_x, 100)
            plt.plot(self.samples, np.zeros(np.shape(self.samples)),
                     colors['samples']+'+', ms=20, label='Samples')
            (grid, sinterpolated) = self.approximate(x, vb=vb,
                                                     using='samples')
            plt.plot(grid, sinterpolated, color=colors['samples'], lw=2.0,
                        alpha=1.0, linestyle=styles['samples'],
                        label='Samples Interpolated PDF')
            extrema = [min(extrema[0], min_x), max(extrema[1], max_x)]
            if vb:
                print('Plotted samples')

        plt.xlim(extrema[0], extrema[-1])
        plt.legend(fontsize='small')
        plt.xlabel('z')
        plt.ylabel('Probability density')
        plt.savefig(loc, dpi=250)

        return

    def kld(self, limits=None, dx=0.01):
        """
        Calculates Kullback-Leibler divergence of quantile approximation from truth.

        Parameters
        ----------
        limits: tuple of floats, optional
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
        print('This function is deprecated; use `qp.utils.calculate_kl_divergence`.')
        return
        # if self.truth is None:
        #     print('Truth not available for comparison.')
        #     return
        # else:
        #     if limits is None:
        #         limits = self.limits
        #     KL = qp.utils.calculate_kl_divergence(self, self, limits=limits, dx=dx)
        #     return(KL)

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
        print('This function is deprecated; use `qp.utils.calculate_rmse`.')
        return
        # if self.truth is None:
        #     print('Truth not available for comparison.')
        #     return
        # else:
        #     RMS = qp.utils.calculate_rms(self, self, limits=limits, dx=dx)
        #     return(RMS)
