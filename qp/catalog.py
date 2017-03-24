import multiprocessing as mp
# import numpy as np
# import scipy.interpolate as spi
# import matplotlib.pyplot as plt

import qp

class catalog(object):

    def __init__(self, pdfs=[], nps=1, nbs=100, vb=True):
        """
        An object containing an ensemble of qp.PDF objects

        Parameters
        ----------
        pdfs: list, optional
            list of qp.PDF objects
        nps: int, optional
            number of processors to use
        nbs: int, optional
            default number of parameters to use in all approximations
        vb: boolean, optional
            report on progress to stdout?
        """
        self.pdfs = pdfs
        self.n_pdfs = len(pdfs)

        self.n_procs = nps
        self.pool = mp.Pool(self.nps)

        self.n_params = nbs

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
        return

    def sample(self, N=None, using='truth'):
        """
        Returns array of samples from all qp.PDF objects in the catalog

        Parameters
        ----------
        N: int, optional
            number of samples to take from each qp.PDF objects
        using: string, optional
            format from which to take samples ('histogram', 'quantiles', 'truth')

        Returns
        -------
        self.samples: numpy.ndarray
            array
        """
        if N is None:
            N = self.nbs
        sample = lambda pdf: pdf.sample(N, using=using, vb=False)
        self.pool.join()
        self.samples = np.array(self.pool.map(sample, pdfs))
        self.pool.close()
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
