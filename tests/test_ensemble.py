"""
Unit tests for PDF class
"""
import sys
import os
import numpy as np, scipy.stats as sps
import unittest
import qp

from data import *


class EnsembleTestCase(unittest.TestCase):

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self.ens_n = build_ensemble(GEN_TEST_DATA['norm'])
        
    def tearDown(self):
        "Clean up any mock data files created by the tests."
        pass


    def _run_ensemble_funcs(self, ens, xpts):
        """Run the test for a practicular class"""

        pdfs = ens.pdf(xpts)
        cdfs = ens.cdf(xpts)
        logpdfs = ens.logpdf(xpts)
        logcdfs = ens.logcdf(xpts)

        assert np.allclose(np.log(pdfs), logpdfs)
        assert np.allclose(np.log(cdfs), logcdfs)
                
        binw = xpts[1:] - xpts[0:-1]
        check_cdf = ((pdfs[:,0:-1] + pdfs[:,1:]) * binw /2).cumsum(axis=1) - cdfs[:,1:]
        assert np.allclose(check_cdf, 0, atol=5e-2)

        hist = ens.histogramize(xpts)[1]
        hist_check = ens.frozen.histogramize(xpts)[1]
        assert np.allclose(hist, hist_check, atol=1e-5)

        ppfs = ens.ppf(QUANTS)
        check_ppf = ens.cdf(ppfs) - QUANTS
        assert np.allclose(check_ppf, 0, atol=1e-9)

        sfs = ens.sf(xpts)
        check_sf = sfs + cdfs
        assert np.allclose(check_sf, 1, atol=1e-5)
        
        isfs = ens.isf(QUANTS)
        check_isf = ens.cdf(ppfs) + QUANTS[::-1]
        assert np.allclose(check_isf, 1, atol=1e-5)


        samples = ens.rvs(size=1000)
        assert samples.shape[0] == ens.frozen.npdf
        assert samples.shape[1] == 1000
        
        median = ens.median()
        mean = ens.mean()
        var = ens.var()
        std = ens.std()
        entropy = ens.entropy()
        stats = ens.stats()
        
        integral = ens.integrate(limits=(ens.gen_obj.a, ens.gen_obj.a))
        interval = ens.interval(0.05)

        for N in range(4):
            check_moment = ens.moment_partial(N, limits=(-5, 5)) - qp.calculate_moment(ens, N, limits=(-5.,5))
            assert np.allclose(check_moment, 0, atol=1e-2)

            sps_moment = ens.moment(N)
            #pmf = ens.pmf(N)
            #logpmf = ens.logpmf(N)

            
    def test_norm(self):
        key = 'norm'
        test_data = GEN_TEST_DATA[key]
        self.ens_n = build_ensemble(test_data)
        assert hasattr(self.ens_n, 'gen_func')
        assert isinstance(self.ens_n.gen_obj, qp.norm_gen)
        assert 'loc' in self.ens_n.frozen.kwds
        self._run_ensemble_funcs(self.ens_n, test_data['test_xvals'])
            
    def test_hist(self):
        key = 'hist'
        test_data = GEN_TEST_DATA[key]
        self.ens_n = build_ensemble(test_data)
        assert isinstance(self.ens_n.gen_obj, qp.hist_rows_gen)
        self._run_ensemble_funcs(self.ens_n, test_data['test_xvals'])

        
        
                                           

if __name__ == '__main__':
    unittest.main()
