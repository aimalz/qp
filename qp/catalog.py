import pandas as pd

import qp

class Catalog(object):

    def __init__(self, ensemble, vb=True):
        """
        An object for storing content produced by the qp.Ensemble

        Parameters
        ----------
        ensemble: qp.Ensemble object
            the qp.Ensemble to which the qp.Catalog is linked
        vb: boolean, optional
            report on progress to stdout?

        Notes
        -----
        The back-end to a qp.Ensemble object that stores and recalls
        parameter values.  The goal is for this to be modular, so it can be
        replaced by more efficient storage later on.  The motivation is that
        qp.Ensemble objects with many parametrizations and/or
        parameters may have very large qp.PDF objects, which will be a problem
        when there are many members in the qp.Ensemble!  This version will be a
        simple pandas.DataFrame, but a true database will eventually replace
        that.
        """
        self.ensemble = ensemble
        self.catalog = pd.DataFrame(columns = self.ensemble.parametrizations)

    def merge(self, other, vb=True):
        """
        Combines two qp.Catalog objects

        Parameters
        ----------
        other: qp.Catalog
            the qp.Catalog to be merged in

        Notes
        -----
        At some point this will be generalized to `other` being a list of
        qp.Catalog objects.
        """
        if vb:
            print('in: ' + str(self.catalog.columns)) + ' + ' + str(other.catalog.columns))

        out_catalog = self.catalog.combine_first(other)

        if vb:
            print('out: ' + str(out_catalog.columns))

        return(out_catalog)

    def add_pdf(self, pdf, vb=True):
        """
        Adds the contents of one qp.PDF object to the catalog
        """
