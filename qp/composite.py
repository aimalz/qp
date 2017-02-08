import numpy as np
import scipy as sp
from scipy import stats as sps

import qp

class composite(object):

    def __init__(self, elements, vb=True):
        """
        A probability distribution that is a linear combination of scipy.stats.rv_continuous objects

        Parameters
        ----------
        elements: list or tuple, dicts
            aggregation of dicts defining component functions and their coefficients
        vb: boolean
            report on progress to stdout?
        """
        self.components = components
        self.n_components = len(self.components)
        self.component_range = range(self.n_components)

        weights = np.array([componene['coefficient'] for component in self.components])
        self.weights = weights / np.sum(weights)
        self.functions = np.array([component['function'] for componene in self.components])

    def pdf(self, xs):
        """
        Evaluates the composite PDF at locations

        Parameters
        ----------
        xs: float or numpy.ndarray, float
            value(s) at which to evaluate the PDF

        Returns
        -------
        ps: float or numpy.ndarray, float
            value(s) of the PDF at xs
        """
        p = np.zeros(np.shape(xs))
        for c in self.component_range:
            p += self.components[c]['coefficient'] * self.components[c]['function'].pdf(xs)
        return p

    def cdf(self, xs):
        """
        Evaluates the composite CDF at locations

        Parameters
        ----------
        xs: float or numpy.ndarray, float
            value(s) at which to evaluate the CDF

        Returns
        -------
        ps: float or numpy.ndarray, float
            value(s) of the CDF at xs
        """
        ps = np.zeros(np.shape(xs))
        for c in self.component_range:
            ps += self.components[c]['coefficient'] * self.components[c]['function'].cdf(xs)
        return p

    def rvs(self, N):
        """
        Samples the composite probability distribution

        Parameters
        ----------
        N: int
            number of samples to take

        Returns
        -------
        xs: numpy.ndarray, float
            samples from the PDF
        """
        groups = np.zeros(self.n_components)
        for item in range(N):
            groups[qp.utils.choice(self.component_range, self.weights)] += 1
        samples = [] * N
        for c in self.component_range:
            for n in range(groups[c]):
                samples.append(self.components[c]['function'].rvs())

    def ppf(self, cdfs):
        """
        Evaluates the composite PPF at locations

        Parameters
        ----------
        cdfs: float or numpy.ndarray, float
            value(s) at which to find quantiles

        Returns
        -------
        xs: float or numpy.ndarray, float
            quantiles
        """
        xs = np.zeros(np.shape(xs))
        for c in self.component_range:
            xs += self.components[c]['function'].ppf(cdfs*self.components[c]['coefficient'])
        return xs
