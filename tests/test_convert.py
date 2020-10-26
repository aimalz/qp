"""
Unit tests for PDF class
"""
import sys
import os
import numpy as np, scipy.stats as sps
import unittest
import qp

from data import *


class ConvertTestCase(unittest.TestCase):

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self.ens_n = build_ensemble(GEN_TEST_DATA['norm'])
        
    def tearDown(self):
        "Clean up any mock data files created by the tests."
        pass


    def _run_convert(self, test_data, **kwargs):
        """Run the test for a practicular class"""

        gen_class = test_data['gen_class']
        try:
            ensemble = self.ens_n.convert_to(gen_class, **test_data['convert_data'])
        except Exception as msg:
            raise ValueError("Failed to make %s %s" % (test_data['convert_data'], msg))
        xpts = test_data['test_xvals']

        diffs = self.ens_n.pdf(xpts) - ensemble.pdf(xpts)
        if not np.allclose(diffs, 0, atol=kwargs.get('atol_diff', 1e-1)):
            pass
            #raise ValueError("%.2e %.2e" % (diffs.min(), diffs.max()))

        ens2 = qp.qp_convert(self.ens_n.frozen, gen_class, **test_data['convert_data'])
        diffs2 = ensemble.pdf(xpts) - ens2.pdf(xpts)        
        assert np.allclose(diffs2, 0, atol=kwargs.get('atol_diff2', 1e-5))

        
    def test_convert_to_interp(self):
        key = 'interp'
        self._run_convert(GEN_TEST_DATA[key])
        
    def test_convert_tospline(self):
        key = 'spline'
        self._run_convert(GEN_TEST_DATA[key])

    def test_convert_tohist(self):
        key = 'hist'
        self._run_convert(GEN_TEST_DATA[key], atol_diff=HIST_TOL)

    def test_convert_toquant(self):
        key = 'quant'
        self._run_convert(GEN_TEST_DATA[key])

    def test_convert_tokde(self):
        key = 'kde'
        # Using a different set of samples, so let the tolerance be larger
        self._run_convert(GEN_TEST_DATA[key], atol_diff2=1e-1)
        
    def test_convert_tomixmod(self):
        key = 'mixmod'
        self._run_convert(GEN_TEST_DATA[key])

    def test_convert_kde_to_hist(self):
        test_data = GEN_TEST_DATA['kde']
        ens_kde = build_ensemble(test_data)
        xpts = test_data['test_xvals']
        ens_hist = ens_kde.convert_to(qp.hist_rows_gen, bins=XBINS, size=100)
        diffs = ens_kde.pdf(xpts) - ens_hist.pdf(xpts)
        if not np.allclose(diffs, 0, atol=HIST_TOL):
            raise ValueError("%.2e %.2e %.2e" % (diffs.min(), diffs.max(), HIST_TOL))
        #assert np.allclose(diffs, 0, atol=HIST_TOL)

    def test_convert_toflex(self):
        key = 'flex'
        self._run_convert(GEN_TEST_DATA[key])

        
if __name__ == '__main__':
    unittest.main()
