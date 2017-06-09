import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import psutil
# import scipy.interpolate as spi
# import matplotlib.pyplot as plt

import qp

class Ensemble(object):

    def __init__(self, N, truth=None, quantiles=None, histogram=None,
                 gridded=None, samples=None, scheme='linear', vb=True):
        """
        An object comprised of many qp.PDF objects to efficiently perform
        operations on all of them

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
            report on progress to stdout?

        Notes
        -----
        There are so many more things I would like qp.Ensemble to be able to do!
        Currently, this is written for backwards compatibility and will have to
        be updated when the qp.PDF object is upgraded.
        """
        self.n_procs = psutil.cpu_count() - 1
        self.pool =Pool(self.n_procs)

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

        self.scheme = scheme

        if vb and self.truth is None and self.quantiles is None and self.histogram is None and self.gridded is None and self.samples is None:
            print 'Warning: initializing an Ensemble object without inputs'
            return

        self.logfilename = 'logfile.txt'

        self.make_pdfs()

    def make_pdfs(self):
        """
        Makes a list of qp.PDF objects based on input
        """
        def make_pdfs_helper(i):
            with open(self.logfilename, 'a') as logfile:
                logfile.write('making pdf '+str(i)+'\n')
            return qp.PDF(truth=self.truth[i], quantiles=self.quantiles[i],
                            histogram=self.histogram[i],
                            gridded=self.gridded[i], samples=self.samples[i],
                            scheme=self.scheme, vb=False)

        self.pdfs = self.pool.map(make_pdfs_helper, self.pdf_range)

        return

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
            def sample_helper(i):
                with open(self.logfilename, 'a') as logfile:
                    logfile.write('sampling pdf '+str(i)+'\n')
                return self.pdfs[i].sample(N=N, infty=infty, using=using, vb=False)

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
            report on progress to stdout?

        Returns
        -------
        self.quantiles: ndarray, tuple, ndarray, float
            array of tuples of the CDF values and the quantiles for each PDF
        """
        def quantize_helper(i):
            with open(self.logfilename, 'a') as logfile:
                logfile.write('quantizing pdf '+str(i)+'\n')
            return self.pdfs[i].quantize(quants=quants, percent=percent,
                                            N=N, infty=infty, vb=False)

        self.quantiles = self.pool.map(quantize_helper, self.pdf_range)

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
            Report on progress to stdout?

        Returns
        -------
        self.histogram: ndarray, tuple, ndarray, floats
            Array of pairs of arrays of lengths (N+1, N) containing endpoints
            of bins and values in bins
        """
        def histogram_helper(i):
            with open(self.logfilename, 'a') as logfile:
                logfile.write('histogramizing pdf '+str(i)+'\n')
            return self.pdfs[i].histogramize(binends=binends, N=N,
                                                binrange=binrange, vb=False)

        self.histogram = self.pool.map(histogram_helper, self.pdf_range)

        return self.histogram

    def mix_mod_fit(self, N=5, using=None, vb=True):
        """
        Fits the parameters of a given functional form to an approximation

        Parameters
        ----------
        N: int, optional
            number of components to consider
        using: string, optional
            which existing approximation to use, defaults to first approximation
        vb: boolean
            Report progress on stdout?

        Returns
        -------
        self.mix_mod: list, qp.Composite objects
            list of qp.Composite objects approximating the PDFs

        Notes
        -----
        Currently only supports mixture of Gaussians
        """
        def mixmod_helper(i):
            with open(self.logfilename, 'a') as logfile:
                logfile.write('fitting pdf '+str(i)+'\n')
            return self.pdfs[i].mix_mod_fit(n_components=N, using=using, vb=False)

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
            report on progress to stdout?

        Returns
        -------
        vals: ndarray, ndarray, float
            the values of the PDFs (or their approximations) at the requested
            location(s), of shape (npdfs, nlocs)
        """
        def evaluate_helper(i):
            with open(self.logfilename, 'a') as logfile:
                logfile.write('evaluating pdf '+str(i)+'\n')
            return self.pdfs[i].evaluate(n_components=loc, using=using, vb=False)

        self.evaluated = self.pool.map(evaluate_helper, self.pdf_range)

        return self.evaluated

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
#             report on progress to stdout?
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
#
#     def add_pdfs(self, pdfs, idnos=None, vb=True):
#         """
#         Adds multiple qp.PDF objects to the qp.Ensemble
#
#         Parameters
#         ----------
#         pdfs: list, qp.PDF
#             the list of qp.PDF objects to be added to the qp.Ensemble
#         idnos: list, string
#             the list of corresponding ID numbers for the qp.pdfs
#         vb: boolean, optional
#             report on progress to stdout?
#         """
#
#         self.catalog
#
#     def evaluate(self, loc, using):
#
#     def help_sample(self, sample_container, n, N, using):#remove container stuff
#         """
#         Helper function for sampling the catalog
#
#         Parameters
#         ----------
#         sample_container: list
#             where the samples are being stored
#         n: int
#             catalog index of PDF to be sampled
#         N: int
#             number of samples to take
#         using: string
#             parametrization/approximation to use
#         """
#         samples = self.pdfs[n].sample(N, using=using, vb=False)
#         #sample_container[n] = samples
#         return
#
#     def sample(self, N=None, using=None):
#         """
#         Returns array of samples from all qp.PDF objects in the catalog
#
#         Parameters
#         ----------
#         N: int, optional
#             number of samples to take from each qp.PDF objects
#         using: string, optional
#             format from which to take samples ('histogram', 'quantiles',
#             'truth')
#             if same for entire catalog
#
#         Returns
#         -------
#         self.samples: numpy.ndarray
#             array of samples for all qp.PDF objects in catalog
#         """
#         if N is None:
#             N = self.n_params
#         self.samples = [None] * self.n_pdfs
#         do_sample = lambda n: self.help_sample(self.samples, n, N, using)
#         #self.pool.join()
#         self.pool.map(do_sample, self.pdf_indices)
#         #self.pool.close()
#         #self.pool.join()
#         #self.pool.close()
#         self.samples = np.array(self.samples)
#         return self.samples
#
#     def quantize(self, N=None):
#         """
#         Makes quantiles of all qp.PDF objects in the catalog
#
#         Parameters
#         ----------
#         N: int, optional
#             number of samples to take from each qp.PDF objects
#         using: string, optional
#             format from which to take samples ('histogram', 'quantiles',
#             'truth')
#
#         Returns
#         -------
#         self.quantiles: dict of numpy.ndarrays
#             input cdfs, array of output quantiles
#         """
#
#         return
#
#     def approximate(self, format, **kwargs):
#         """
#         Produces an array of the entire catalog in some format
#
#         Parameters
#         ----------
#         format: string
#             currently supports 'quantiles', 'histogram', 'samples'
#         **kwargs: dictionary
#             dictionary of arguments for format
#
#         Returns
#         -------
#         output: dictionary
#
#         """
#
#         return
#
#     def select(self, function):
#         """
#         """
