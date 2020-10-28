"""Implemenation of an ensemble of distributions"""

import os
import numpy as np
# import psutil
#import timeit

from astropy.table import Table


from .dict_utils import slice_dict, print_dict_shape

from .metrics import quick_kld, quick_rmse, quick_moment

from .persistence import get_qp_reader


class Ensemble:
    """An object comprised of many qp.PDF objects to efficiently perform operations on all of them"""
    def __init__(self, gen_func, data):
        """Class constructor

        Parameters
        ----------
        gen_func : `function`
            Function that creates generic distribution object
        data : `dict`
            Dictionary with data used to construct the ensemble

        """
        #start_time = timeit.default_timer()
        self._gen_func = gen_func
        self._frozen = self._gen_func(**data)
        self._gen_obj = self._frozen.dist
        self._gen_class = type(self._gen_obj)

        self._gridded = None
        self._samples = None


    def __getitem__(self, key):
        """Build a `scipy.rv_frozen` object for a sub-set of the PDFs in this ensemble

        Parameter
        ---------
        key : `int` or `slice`
            Used to slice the data to pick out one PDF from this ensemble

        Returns
        -------
        pdf : `scipy.rv_frozen`
            The distribution for the requeseted element or slide
        """
        red_data = slice_dict(self._gen_obj.objdata, key)
        red_data.update(slice_dict(self._frozen.kwds, key))
        return self._gen_obj(**red_data)

    @property
    def gen_func(self):
        """Return the function used to create the distribution object for this ensemble"""
        return self._gen_func

    @property
    def gen_class(self):
        """Return the class used to generate distributions for this ensemble"""
        return self._gen_class

    @property
    def gen_obj(self):
        """Return the `scipy.stats.rv_continuous` object that generates distributions for this ensemble"""
        return self._gen_obj

    @property
    def frozen(self):
        """Return the `scipy.stats.rv_frozen` object that encapsultes the distributions for this ensemble"""
        return self._frozen

    def convert_to(self, gen_class, method=None, **kwargs):
        """Convert a distribution or ensemble

        Parameters
        ----------
        gen_class :  `scipy.stats.rv_continuous or qp.ensemble`
            Class to convert to
        method : `str`
            Optional argument to specify a non-default conversion algorithm
        kwargs : keyword arguments are passed to the output class constructor

        Returns
        -------
        ens : `qp.Ensemble`
            Ensemble of pdfs yype class_to using the data from this object
        """
        return gen_class.convert_from(self, method, **kwargs)

    def metadata(self):
        """Return the metadata for this ensemble

        Returns
        -------
        metadata : `dict`
            The metadata

        Notes
        -----
        Metadata are elements that are the same for all the PDFs in the ensemble
        These include the name and version of the PDF generation class
        """

        dd = {}
        dd.update(self._gen_obj.metadata)
        return dd

    def objdata(self):
        """Return the object data for this ensemble

        Returns
        -------
        objdata : `dict`
            The object data

        Notes
        -----
        Object data are elements that differ for each PDFs in the ensemble
        """

        dd = {}
        dd.update(self._frozen.kwds)
        dd.pop('row', None)
        dd.update(self._gen_obj.objdata)
        return dd

    def build_tables(self):
        """Build and return `astropy.Table` objects for the meta data and object data
        for this ensemble

        Returns
        -------
        meta : `astropy.Table`
            Table with the meta data
        data : `astropy.Table`
            Table with the object data
        """
        try:
            meta = Table(self.metadata())
        except ValueError as exep: #pragma : no cover
            print_dict_shape(self.metadata())
            raise ValueError from exep
        try:
            data = Table(self.objdata())
        except ValueError as exep: #pragma : no cover
            print_dict_shape(self.objdata())
            raise ValueError from exep
        return dict(meta=meta, data=data)

    def gridded(self, grid):
        """Build, cache are return the PDF values at grid points

        Parameters
        ----------
        grid : array-like
            The grid points

        Returns
        -------
        gridded : (grid, pdf_values)

        Notes
        -----
        This first comparse grid to the cached value, if they match it returns
        the cached value
        """
        if self._gridded is None or not np.array_equal(self._gridded[0], grid):
            self._gridded = (grid, self.pdf(grid))
        return self._gridded

    def write_to(self, filename):
        """Save this ensemble to a file

        Parameters
        ----------
        filename : `str`

        Notes
        -----
        This will actually write two files, one for the metadata and one for the object data

        This uses `astropy.Table` to write the data, so any filesuffix that works for
        `astropy.Table.write` will work here.
        """
        basename, ext = os.path.splitext(filename)
        meta_ext = "_meta%s" % ext
        meta_filename = basename + meta_ext

        tables = self.build_tables()
        tables['meta'].write(meta_filename, overwrite=True)
        tables['data'].write(filename, overwrite=True)


    @classmethod
    def read_from(cls, filename):
        """Read this ensemble from a file

        Parameters
        ----------
        filename : `str`

        Notes
        -----
        This will actually read two files, one for the metadata and one for the object data

        This uses `astropy.Table` to write the data, so any filesuffix that works for
        `astropy.Table.write` will work here.

        This will use information in the meta data to figure out how to construct the data
        need to build the ensemble.
        """
        basename, ext = os.path.splitext(filename)
        meta_ext = "_meta%s" % ext
        meta_filename = basename + meta_ext

        md_table = Table.read(meta_filename)
        data_table = Table.read(filename)

        data_dict = {}
        for col in md_table.columns:
            col_data = md_table[col].data
            if len(col_data.shape) > 1:
                col_data = np.squeeze(col_data)

            if col_data.size == 1:
                col_data = col_data[0]

            if isinstance(col_data, bytes):
                col_data = col_data.decode()
            data_dict[col] = col_data


        for col in data_table.columns:
            col_data = data_table[col].data
            if len(col_data.shape) < 2:
                data_dict[col] = np.expand_dims(data_table[col].data, -1)
            else:
                data_dict[col] = col_data

        pdf_name = data_dict.pop('pdf_name')
        pdf_version = data_dict.pop('pdf_version')
        class_to = get_qp_reader(pdf_name, pdf_version)
        return cls(class_to.create, data=data_dict)


    def pdf(self, x):
        """
        Evaluates the probablity density function for the whole ensemble

        Parameters
        ----------
        x: float or ndarray, float
            location(s) at which to do the evaluations

        Returns
        -------
        """
        return self._frozen.pdf(x)

    def logpdf(self, x):
        """
        Evaluates the log of the probablity density function for the whole ensemble

        Parameters
        ----------
        x: float or ndarray, float
            location(s) at which to do the evaluations

        Returns
        -------
        """
        return self._frozen.logpdf(x)


    def cdf(self, x):
        """
        Evaluates the cumalative distribution function for the whole ensemble

        Parameters
        ----------
        x: float or ndarray, float
            location(s) at which to do the evaluations

        Returns
        -------
        """
        return self._frozen.cdf(x)

    def logcdf(self, x):
        """
        Evaluates the log of the cumalative distribution function for the whole ensemble

        Parameters
        ----------
        x: float or ndarray, float
            location(s) at which to do the evaluations

        Returns
        -------
        """
        return self._frozen.logcdf(x)

    def ppf(self, q):
        """
        Evaluates all the PPF of the distribution

        Parameters
        ----------
        q: float or ndarray, float
            location(s) at which to do the evaluations

        Returns
        -------
        """
        return self._frozen.ppf(q)


    def sf(self, q):
        """
        Evaluates the survival fraction of the distribution

        Parameters
        ----------
        x: float or ndarray, float
            (s) at which to evaluate the pdfs

        Returns
        -------
        """
        return self._frozen.sf(q)

    def isf(self, q):
        """
        Evaluates the inverse of the survival fraction of the distribution

        Parameters
        ----------
        x: float or ndarray, float
            (s) at which to evaluate the pdfs

        Returns
        -------
        """
        return self._frozen.isf(q)

    def rvs(self, size=None, random_state=None):
        """
        Generate samples from this ensmeble

        Parameters
        ----------
        size: int
            number of samples to return

        Returns
        -------
        """
        return self._frozen.rvs(size=(self._frozen.npdf, size), random_state=random_state)


    def stats(self, moments='mv'):
        """
        Retrun the stats for this ensemble

        Parameters
        ----------
        moments: `str`
            Which moments to include

        Returns
        -------
        """
        return self._frozen.stats(moments=moments)

    def median(self):
        """ Return the medians for this ensemble """
        return self._frozen.median()

    def mean(self):
        """ Return the means for this ensemble """
        return self._frozen.mean()

    def var(self):
        """ Return the variences for this ensemble """
        return self._frozen.var()

    def std(self):
        """ Return the standard deviations for this ensemble """
        return self._frozen.std()

    def moment(self, n):
        """ Return the nth moments for this ensemble """
        return self._frozen.moment(n)

    def entropy(self):
        """ Return the entropy for this ensemble """
        return self._frozen.entropy()

    #def pmf(self, k):
    #    """ Return the kth pmf for this ensemble """
    #    return self._frozen.pmf(k)

    #def logpmf(self, k):
    #    """ Return the log of the kth pmf for this ensemble """
    #    return self._frozen.logpmf(k)

    def interval(self, alpha):
        """ Return the intervals corresponding to a confidnce level of alpha for this ensemble"""
        return self._frozen.interval(alpha)


    def histogramize(self, bins):
        """
        Computes integrated histogram bin values for all PDFs

        Parameters
        ----------
        bins: ndarray, float, optional
            Array of N+1 endpoints of N bins

        Returns
        -------
        self.histogram: ndarray, tuple, ndarray, floats
            Array of pairs of arrays of lengths (N+1, N) containing endpoints
            of bins and values in bins
        """
        return self._frozen.histogramize(bins)

    def integrate(self, limits):
        """
        Computes the integral under the ensemble of PDFs between the given limits.

        Parameters
        ----------
        limits: numpy.ndarray, tuple, float
            limits of integration, may be different for all PDFs in the ensemble
        using: string
            parametrization over which to approximate the integral
        dx: float, optional
            granularity of integral

        Returns
        -------
        integral: numpy.ndarray, float
            value of the integral
        """
        return self.cdf(limits[1]) - self.cdf(limits[0])


    def mix_mod_fit(self, comps=5): #pragma: no cover
        """
        Fits the parameters of a given functional form to an approximation

        Parameters
        ----------
        comps: int, optional
            number of components to consider
        using: string, optional
            which existing approximation to use, defaults to first approximation
        vb: boolean
            Report progress

        Returns
        -------
        self.mix_mod: list, qp.Composite objects
            list of qp.Composite objects approximating the PDFs

        Notes
        -----
        Currently only supports mixture of Gaussians
        """
        raise NotImplementedError("mix_mod_fit %i" % comps)


    def moment_partial(self, n, limits, dx=0.01):
        """ Return the nth moments for this over a particular range"""
        D = int((limits[-1] - limits[0]) / dx)
        grid = np.linspace(limits[0], limits[1], D)
        # dx = (limits[-1] - limits[0]) / (D - 1)

        P_eval = self.gridded(grid)[1]
        grid_to_n = grid**n
        return quick_moment(P_eval, grid_to_n, dx)


    def kld(self, other, limits, dx=0.01):
        """
        Calculates the KLD for each PDF in the ensemble
        Parameters
        ----------
        other: `qp.ensemble`
            Other ensemble
        limits: tuple of floats, optional
            endpoints of integration interval in which to calculate KLD
        dx: float
            resolution of integration grid

        Returns
        -------
        klds: numpy.ndarray, float
            KLD values of each PDF under the using approximation relative to the truth
        """
        D = int((limits[-1] - limits[0]) / dx)
        grid = np.linspace(limits[0], limits[1], D)
        # dx = (limits[-1] - limits[0]) / (D - 1)

        P_eval = other.gridded(grid)[1]
        Q_eval = self.gridded(grid)[1]
        def kld_helper(p_row, q_row):
            return quick_kld(p_row, q_row, dx)
        vv = np.vectorize(kld_helper)
        klds = vv(P_eval, Q_eval)
        return klds



    def rmse(self, other, limits, dx=0.01):
        """
        Calculates the RMSE for each PDF in the ensemble
        Parameters
        ----------
        other: `qp.ensemble`
            Other ensemble
        limits: tuple of floats, optional
            endpoints of integration interval in which to calculate KLD
        dx: float
            resolution of integration grid

        Returns
        -------
        rmses: numpy.ndarray, float
            KLD values of each PDF under the using approximation relative to the truth
        """
        D = int((limits[-1] - limits[0]) / dx)
        grid = np.linspace(limits[0], limits[1], D)
        # dx = (limits[-1] - limits[0]) / (D - 1)

        P_eval = other.gridded(grid)[1]
        Q_eval = self.gridded(grid)[1]
        def rmse_helper(p_row, q_row):
            return quick_rmse(p_row, q_row, D)
        vv = np.vectorize(rmse_helper)
        rmses = vv(P_eval, Q_eval)
        return rmses


    # def stack(self, loc, using, vb=True):
    #     """
    #     Produces an average of the PDFs in the ensemble
    #
    #     Parameters
    #     ----------
    #     loc: ndarray, float or float
    #         location(s) at which to evaluate the PDFs
    #     using: string
    #         which parametrization to use for the approximation
    #     vb: boolean
    #         report on progress
    #
    #     Returns
    #     -------
    #     self.stacked: tuple, ndarray, float
    #         pair of arrays for locations where approximations were evaluated
    #         and the values of the stacked PDFs at those points
    #
    #     Notes
    #     -----
    #     Stacking refers to taking the sum of PDFs evaluated on a shared grid and normalizing it such that it integrates to unity.  This is equivalent to calculating an average probability (based on the PDFs in the ensemble) over the grid.  This probably should be done in a script and not by qp!  The right way to do it would be to call qp.Ensemble.evaluate() and sum those outputs appropriately.
    #     TO DO: make this do something more efficient for mixmod, grid, histogram, samples
    #     TO DO: enable stacking on irregular grid
    #     """
    #     loc_range = max(loc) - min(loc)
    #     delta = loc_range / len(loc)
    #     evaluated = self.evaluate(loc, using=using, norm=True, vb=vb)
    #     stack = np.mean(evaluated[1], axis=0)
    #     stack /= np.sum(stack) * delta
    #     assert(np.isclose(np.sum(stack) * delta, 1.))
    #     self.stacked[using] = (evaluated[0], stack)
    #     return self.stacked

# Note: A copious quantity of commented code has been removed in this commit!
# For future reference, it can still be found here:
#  https://github.com/aimalz/qp/blob/d8d145af9514e29c76e079e869b8b4923f592f40/qp/ensemble.py
# Critical additions still remain.  Metrics of individual qp.PDF objects collected in aggregate over a qp.Ensemble are still desired.
