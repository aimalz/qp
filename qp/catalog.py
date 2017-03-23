import multiprocessing as mp
# import numpy as np
# import scipy.interpolate as spi
# import matplotlib.pyplot as plt

import qp

class catalog(object):

    def __init__(self, pdfs=[], nps=mp.cpu_count()-1, vb=True):
        """
        An object containing an ensemble of qp.PDF objects

        Parameters
        ----------
        pdfs: list, optional
            list of qp.PDF objects
        vb: boolean
            report on progress to stdout?
        """
        self.pdfs = pdfs
        self.nps = nps
        self.pool = mp.Pool(self.nps)

    def add(self, pdfs):
        """
        Adds qp.PDF objects to an existing qp.catalog object

        Parameters
        ----------
        pdfs: list
            list of qp.PDF objects
        """
        self.pdfs = self.pdfs + pdfs
        return

    def sample(self, N, using):
        """
        Returns array of samples from all qp.PDF objects in the catalog

        Parameters
        ----------
        N
        """
        output = np.array(self.pool.map(lambda pdf: pdf.sample(N, using=using, vb=False), pdfs))

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
