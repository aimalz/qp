import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import psutil
import os
# import sqlalchemy
import scipy.interpolate as spi
import matplotlib.pyplot as plt

import qp

class Ensemble(object):

    def __init__(self, N, pdfs=None, truth=None, quantiles=None, histogram=None, gridded=None, samples=None, scheme='linear', vb=True, procs=None):# where='ensemble.db', procs=None):#
        """
        Creates an object comprised of many qp.PDF objects to efficiently
        perform operations on all of them

        Parameters
        ----------
        N: int
            number of pdfs in the ensemble
        truth: list of scipy.stats.rv_continuous objects or qp.composite objects
            , optional
            List of length (npdfs) containing the continuous, parametric forms of the PDFs
        quantiles: tuple of ndarrays, optional
            Pair of arrays of lengths (nquants) and (npdfs, nquants) containing
            shared CDF values and quantiles for each pdf
        histogram: tuple of ndarrays, optional
            Pair of arrays of lengths (nbins+1) and (npdfs, nbins) containing
            shared endpoints of bins and values in bins for each pdf
        gridded: tuple of ndarrays, optional
            Pair of arrays of lengths (npoints) and (npdfs, npoints) containing
            shared points at which pdfs are evaluated and the values of each
            pdf at those points
        samples: ndarray, optional
            Array of size (npdfs, nsamples) containing sampled values
        scheme: string, optional
            name of interpolation scheme to use.
        vb: boolean, optional
            report on progress
        where: string
            path to file corresponding to Ensemble, presumed to not yet exist
        procs: int, optional
            limit the number of processors used, otherwise use all available
        """
        if procs is not None:
            self.n_procs = procs
        else:
            self.n_procs = psutil.cpu_count()
        self.pool = Pool(self.n_procs)

        self.n_pdfs = N
        self.pdf_range = range(N)

        if truth is None:
            self.truth = [None] * N
        else:
            self.truth = truth
        if samples is None:
            self.samples = [None] * N
        else:
            self.samples = samples
        if quantiles is None:
            self.quantiles = [None] * N
        else:
            self.quantiles = [(quantiles[0], quantiles[1][i]) for i in self.pdf_range]
        if histogram is None:
            self.histogram = [None] * N
        else:
            self.histogram = [(histogram[0], histogram[1][i]) for i in self.pdf_range]
        if gridded is None:
            self.gridded = [None] * N
        else:
            self.gridded = [(gridded[0], gridded[1][i]) for i in self.pdf_range]
        self.mix_mod = None
        self.evaluated = None

        self.scheme = scheme

        if vb and self.truth is None and self.quantiles is None and self.histogram is None and self.gridded is None and self.samples is None:
            print 'Warning: initializing an Ensemble object without inputs'
            return

        self.make_pdfs()

        # self.logfilename = 'logfile.txt'

        self.stacked = {}

        # self.where = where
        # if os.path.isfile(self.where):
        #     #something like this, uses connection?
        #     self.engine = self.read(self.where)
        # else:
        #     self.engine = sqlalchemy.create_engine('sql:///'+self.where)
        #
        # self.metadata = MetaData(bind=self.engine)
        #
        # parametrizations_table =Table('parametrizations', metadata,)

    #
    #
    # def read(self):
    #
    # def write(self):
    #
    # def make_db(self, where):
    #     """
    #     Makes a fresh set of ensemble tables
    #     """
    #     return
    #
    # def __add__(self, ensemble):
    #
    # def add_PDFs(self, PDFs):
    #     """
    #     Adds qp.PDF objects to the ensemble
    #
    #     Parameters
    #     ----------
    #     PDFs: list, qp.PDF object
    #         list of PDF objects to add to the ensemble
    #     """
    #     pdf_range = range(len(PDFs))
    #     def add_one(i):
    #
    #
    #     self.pdfs = self.pool.map(add_one, pdf_range)
    #
    #     return(self)
    #
    # def add_parameterization(self, type, parameters):
    #

    def make_pdfs(self):
        """
        Makes a list of qp.PDF objects based on input
        """
        def make_pdfs_helper(i):
            # with open(self.logfilename, 'wb') as logfile:
            #     logfile.write('making pdf '+str(i)+'\n')
            return qp.PDF(truth=self.truth[i], quantiles=self.quantiles[i],
                            histogram=self.histogram[i],
                            gridded=self.gridded[i], samples=self.samples[i],
                            scheme=self.scheme, vb=False)

        self.pdfs = self.pool.map(make_pdfs_helper, self.pdf_range)
        self.pdf_range = range(len(self.pdfs))

        return

    def sample(self, samps=100, infty=100., using=None, vb=True):
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
            report on progress

        Returns
        -------
        samples: ndarray
            array of sampled values
        """
        def sample_helper(i):
            # with open(self.logfilename, 'wb') as logfile:
            #     logfile.write('sampling pdf '+str(i)+'\n')
            return self.pdfs[i].sample(N=samps, infty=infty, using=using, vb=False)

        self.samples = self.pool.map(sample_helper, self.pdf_range)

        return self.samples

    def quantize(self, quants=None, percent=10., N=None, infty=100., vb=True):
        """
        Computes an array of evenly-spaced quantiles for each PDF

        Parameters
        ----------
        quants: ndarray, float, optional
            array of quantile locations as decimals
        percent: float, optional
            the separation of the requested quantiles, in percent
        N: int, optional
            the number of quantiles to compute.
        infty: float, optional
            approximate value at which CDF=1.
        vb: boolean
            report on progress

        Returns
        -------
        self.quantiles: ndarray, tuple, ndarray, float
            array of tuples of the CDF values and the quantiles for each PDF
        """
        def quantize_helper(i):
            # with open(self.logfilename, 'wb') as logfile:
            #     logfile.write('quantizing pdf '+str(i)+'\n')
            return self.pdfs[i].quantize(quants=quants, percent=percent,
                                            N=N, infty=infty, vb=False)

        self.quantiles = self.pool.map(quantize_helper, self.pdf_range)
        self.quantiles = np.swapaxes(np.array(self.quantiles), 0, 1)
        self.quantiles = (self.quantiles[0][0], self.quantiles[1])

        return self.quantiles

    def histogramize(self, binends=None, N=10, binrange=None, vb=True):
        """
        Computes integrated histogram bin values for all PDFs

        Parameters
        ----------
        binends: ndarray, float, optional
            Array of N+1 endpoints of N bins
        N: int, optional
            Number of bins if no binends provided
        binrange: tuple, float, optional
            Pair of values of endpoints of total bin range
        vb: boolean
            Report on progress

        Returns
        -------
        self.histogram: ndarray, tuple, ndarray, floats
            Array of pairs of arrays of lengths (N+1, N) containing endpoints
            of bins and values in bins
        """
        def histogram_helper(i):
            # with open(self.logfilename, 'wb') as logfile:
            #     logfile.write('histogramizing pdf '+str(i)+'\n')
            return self.pdfs[i].histogramize(binends=binends, N=N,
                                                binrange=binrange, vb=False)

        self.histogram = self.pool.map(histogram_helper, self.pdf_range)
        self.histogram = np.swapaxes(np.array(self.histogram), 0, 1)
        self.histogram = (self.histogram[0][0], self.histogram[1])

        return self.histogram

    def mix_mod_fit(self, comps=5, using=None, vb=True):
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
        def mixmod_helper(i):
            # with open(self.logfilename, 'wb') as logfile:
            #     logfile.write('fitting pdf '+str(i)+'\n')
            return self.pdfs[i].mix_mod_fit(n_components=comps, using=using, vb=False)

        self.mix_mod = self.pool.map(mixmod_helper, self.pdf_range)

        return self.mix_mod

    def evaluate(self, loc, using=None, vb=True):
        """
        Evaluates all PDFs

        Parameters
        ----------
        loc: float or ndarray, float
            location(s) at which to evaluate the pdfs
        using: string
            which parametrization to evaluate, defaults to initialization
        vb: boolean
            report on progress

        Returns
        -------
        vals: ndarray, ndarray, float
            the values of the PDFs (or their approximations) at the requested
            location(s), of shape (npdfs, nlocs)
        """
        def evaluate_helper(i):
            # with open(self.logfilename, 'wb') as logfile:
            #     logfile.write('evaluating pdf '+str(i)+'\n')
            return self.pdfs[i].evaluate(loc=loc, using=using, vb=False)
        self.gridded = self.pool.map(evaluate_helper, self.pdf_range)
        self.gridded = np.swapaxes(np.array(self.gridded), 0, 1)
        self.gridded = (self.gridded[0][0], self.gridded[1])

        return self.gridded

    # def approximate(self, points, using=None, scheme=None, vb=True):

    def stack(self, loc, using, vb=True):
        """
        Produces a stack of the PDFs

        Parameters
        ----------
        loc: ndarray, float or float
            location(s) at which to evaluate the PDFs
        using: string
            which parametrization to use for the approximation
        vb: boolean
            report on progress

        Returns
        -------
        self.stacked: tuple, ndarray, float
            pair of arrays for locations where approximations were evaluated
            and the values of the stacked PDFs at those points
        """
        loc_range = max(loc) - min(loc)
        delta = loc_range / len(loc)
        evaluated = self.evaluate(loc, using=using, vb=True)
        stack = np.mean(evaluated[1], axis=0)
        stack /= np.sum(stack) * delta
        assert(np.isclose(np.sum(stack) * delta, 1.))
        self.stacked[using] = (evaluated[0], stack)
        return self.stacked

    def kld(self, stacked=True, limits=(0., 1.), dx=0.01):
        """
        Calculates the KLD for the stacked estimator under different parametrizations

        Parameters
        ----------
        stacked: boolean, optional
            calculate metric on stacked estimator?
        limits: tuple, float, optional

        dx: float, optional

        Returns
        -------

        """
        if self.truth is None:
            print('Truth must be defined for KLD')
            return
        kld = {}
        if stacked == True:
            P = qp.PDF(gridded=self.stack['truth'])
            for est in self.stacked.keys():
                klds[est] = qp.utils.calculate_kl_divergence(P, self.stacked[est], limits=limits, dx=dx)
            return klds
        else:
            print('KLDs of each PDF not yet supported')
            return

    def rms(self, limits=(0., 1.), dx=0.01):
        """
        Calculates the KLD for the stacked estimator under different parametrizations

        Parameters
        ----------
        stacked: boolean, optional
            calculate metric on stacked estimator?
        limits: tuple, float, optional

        dx: float, optional

        Returns
        -------

        """
        if self.truth is None:
            print('Truth must be defined for KLD')
            return
        kld = {}
        if stacked == True:
            P = qp.PDF(gridded=self.stack['truth'])
            for est in self.stacked.keys():
                klds[est] = qp.utils.calculate_kl_divergence(P, self.stacked[est], limits=limits, dx=dx)
            return klds
        else:
            print('KLDs of each PDF not yet supported')
            return
        return

    def plot(self, vb=True):
        return

    def read(self, format, location):
        return

    def write(self, format, location):
        return

# # Total pie in the sky beyond this point!  I'll approach this complication
# # if and when we need to optimize qp further.

# class Ensemble(object):
#
#     def __init__(self, pdfs=None, nprocs=1, vb=True):
#         """
#         An object containing an ensemble of qp.PDF objects
#
#         Parameters
#         ----------
#         pdfs: list, qp.PDF, optional
#             list of qp.PDF objects to turn into an ensemble
#         nprocs: int, optional
#             number of processors to use, defaults to 1
#         vb: boolean, optional
#             report on progress
#
#         Notes
#         -----
#         The qp.Ensemble object is basically a front-end to a qp.Catalog
#         that just stores parameter values.  This will keep the qp.Ensemble from
#         consuming too much memory by holding very large qp.PDF objects.
#         """
#         # all possible parametrizations will be columns
#         self.parametrizations = ['mm', 'gridded', 'histogram', 'quantiles', 'samples']
#         self.metadata = {}
#         for p in self.parametrizations:
#             self.metadata[p] = {}
#
#         if pdfs is None:
#             self.pdfs = []
#             self.pdfids = []
#         else:
#             pdf0 = pdfs[0]
#             pdfids = []
#             for p in self.parametrizations:
#                 pdfids.append()
#                 assert(np.all([]) == True)
#             self.pdfs = pdfs
#
#         self.catalog = qp.Catalog(self)
#
#         self.n_procs = nprocs
#         self.pool = Pool(self.n_procs)
#
#     def __add__(self, other):
#         """
#         Combines two or more ensembles using "+" if parametrizations are the
#         same
#
#         Parameters
#         ----------
#         other: qp.Ensemble
#             the qp.Enseble object to be combined with this one
#
#         Returns
#         -------
#         new_ensemble: qp.Ensemble
#             returns a new ensemble that is the combination of this one and other
#         """
#         if other.metadata != self.metadata:
#             for p in parametrizations:
#                 if other.metadata[p] != {} and self.metadata[p] != {} and other.metadata[p] != self.metadata[p]:
#                     print(p + ' metadata is not the same!')
#             return
#         else:
#             new_catalog = self.catalog.merge(other.catalog)
#             new_ensemble = qp.Ensemble(catalog = new_catalog)
#             return new_ensemble

# from sqlalchemy import Column, Integer, String
# from sqlalchemy.ext.declarative import declarative_base
#
# Base = declarative_base()
#
# class PDF(Base):
#     __tablename__ = 'PDF'
#     id = Column(Integer, primary_key=True)
#
# class parametrization(Base):
#     __tablename__ = 'parametrization'
#     id = Column(Integer, primary_key=True)
#
# class metaparameters(Base):
#     __tablename__ = 'metaparameters'
#     id = Column(Integer, primary_key=True)
#
# class representation(Base):
#     __tablename__ = 'representation'
#     id = Column(Integer, primary_key=True)
