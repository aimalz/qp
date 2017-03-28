import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
#import multiprocessing as mp
# import scipy.interpolate as spi
# import matplotlib.pyplot as plt

import qp

class catalog(object):

    def __init__(self, pdfs=[], nprocs=1, nparams=100, vb=True):
        """
        An object containing an ensemble of qp.PDF objects

        Parameters
        ----------
        pdfs: list, optional
            list of qp.PDF objects
        nprocs: int, optional
            number of processors to use
        nparams: int, optional
            default number of parameters to use in all approximations
        vb: boolean, optional
            report on progress to stdout?
        """
        self.pdfs = pdfs
        self.n_pdfs = len(pdfs)
        self.pdf_indices = range(self.n_pdfs)

        self.n_procs = nprocs
        self.pool = Pool(self.n_procs)
        #self.pool.close()

        self.n_params = nparams

    def add(self, pdfs):
        """
        Adds qp.PDF objects to an existing qp.catalog object

        Parameters
        ----------
        pdfs: list
            list of qp.PDF objects
        """
        self.pdfs = self.pdfs + pdfs
        self.n_pdfs = len(self.pdfs)
        self.pdf_indices = range(self.n_pdfs)
        return

    def help_sample(self, sample_container, n, N, using):
        """
        Helper function for sampling the catalog

        Parameters
        ----------
        sample_container: list
            where the samples are being stored
        n: int
            catalog index of PDF to be sampled
        N: int
            number of samples to take
        using: string
            parametrization/approximation to use
        """
        samples = self.pdfs[n].sample(N, using=using, vb=False)
        sample_container[n] = samples
        return

    def sample(self, N=None, using=None):
        """
        Returns array of samples from all qp.PDF objects in the catalog

        Parameters
        ----------
        N: int, optional
            number of samples to take from each qp.PDF objects
        using: string, optional
            format from which to take samples ('histogram', 'quantiles', 'truth')
            if same for entire catalog

        Returns
        -------
        self.samples: numpy.ndarray
            array of samples for all qp.PDF objects in catalog
        """
        if N is None:
            N = self.n_params
        self.samples = [None] * self.n_pdfs
        do_sample = lambda n: self.help_sample(self.samples, n, N, using)
        #self.pool.join()
        self.pool.map(do_sample, self.pdf_indices)
        #self.pool.close()
        #self.pool.join()
        #self.pool.close()
        self.samples = np.array(self.samples)
        return self.samples

    def quantize(self, N=None):
        """
        Makes quantiles of all qp.PDF objects in the catalog

        Parameters
        ----------
        N: int, optional
            number of samples to take from each qp.PDF objects
        using: string, optional
            format from which to take samples ('histogram', 'quantiles', 'truth')

        Returns
        -------
        self.quantiles: dict of numpy.ndarrays
            array
        """

        return

    def approximate(self, format, **kwargs):
        """
        Produces an array of the entire catalog in some format

        Parameters
        ----------
        format: string
            currently supports 'quantiles', 'histogram', 'samples'
        **kwargs: dictionary
            dictionary of arguments for format

        Returns
        -------
        output: dictionary

        """

        return

    def select(self, function):
        """
        """
