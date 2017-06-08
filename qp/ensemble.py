import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
# import scipy.interpolate as spi
# import matplotlib.pyplot as plt

import qp

class Ensemble(object):

    def __init__(self, catalog=None, nprocs=1, vb=True):
        """
        An object containing an ensemble of qp.PDF objects

        Parameters
        ----------
        catalog
        nprocs: int, optional
            number of processors to use, defaults to 1
        vb: boolean, optional
            report on progress to stdout?

        Notes
        -----
        The qp.Ensemble object is basically a front-end to a qp.Catalog
        that just stores parameter values.  This will keep the qp.Ensemble from
        consuming too much memory by holding very large qp.PDF objects.
        """
        # all possible parametrizations will be columns
        self.parametrizations = ['mm', 'gridded', 'histogram', 'quantiles', 'samples']
        self.metadata = {}
        for p in self.parametrizations:
            self.metadata[p] = {}

        # initialize a qp.Catalog
        if catalog is None:
            self.catalog = qp.Catalog(self)
        else:
            self.catalog = catalog

        self.n_procs = nprocs
        self.pool = Pool(self.n_procs)

    def __add__(self, other):
        """
        Combines two or more ensembles using "+" if parametrizations are the
        same

        Parameters
        ----------
        other: qp.Ensemble
            the qp.Enseble object to be combined with this one

        Returns
        -------
        new_ensemble: qp.Ensemble
            returns a new ensemble that is the combination of this one and other
        """
        if other.metadata != self.metadata:
            for p in parametrizations:
                if other.metadata[p] != {} and self.metadata[p] != {} and other.metadata[p] != self.metadata[p]:
                    print(p + ' metadata is not the same!')
            return
        else:
            new_catalog = self.catalog.merge(other.catalog)
            new_ensemble = qp.Ensemble(catalog = new_catalog)
            return new_ensemble

    def add_pdf(self, pdf, vb=True):
        """
        Adds one qp.PDF object to the qp.Ensemble

        Parameters
        ----------
        pdf: qp.PDF
            the qp.PDF object to be added to the qp.Ensemble
        """

    def evaluate(self, loc, using):

    def help_sample(self, sample_container, n, N, using):#remove container stuff
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
        #sample_container[n] = samples
        return

    def sample(self, N=None, using=None):
        """
        Returns array of samples from all qp.PDF objects in the catalog

        Parameters
        ----------
        N: int, optional
            number of samples to take from each qp.PDF objects
        using: string, optional
            format from which to take samples ('histogram', 'quantiles',
            'truth')
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
            format from which to take samples ('histogram', 'quantiles',
            'truth')

        Returns
        -------
        self.quantiles: dict of numpy.ndarrays
            input cdfs, array of output quantiles
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
