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


    def _run_convert(self, gen_class, test_data, **kwargs):
        """Run the test for a practicular class"""

        try:
            ensemble = self.ens_n.convert_to(gen_class, **test_data['convert_data'])
        except Exception as msg:
            raise ValueError("Failed to make %s %s" % (test_data['convert_data'], msg))
        xpts = test_data['test_xvals']

        diffs = self.ens_n.pdf(xpts) - ensemble.pdf(xpts)
        assert_all_small(diffs, atol=kwargs.get('atol_diff', 1e-1))

        ens2 = qp.qp_convert(self.ens_n.frozen, gen_class, **test_data['convert_data'])
        diffs2 = ensemble.pdf(xpts) - ens2.pdf(xpts)
        assert_all_small(diffs2, atol=kwargs.get('atol_diff2', 1e-5))

        
    def test_convert_to_interp(self):
        key = 'interp'
        self._run_convert(qp.interp_rows_gen, GEN_TEST_DATA[key])
        
    def test_convert_tospline(self):
        key = 'spline'
        self._run_convert(qp.spline_rows_gen, GEN_TEST_DATA[key])

    def test_convert_tospline_samples(self):
        key = 'spline_kde'
        self._run_convert(qp.spline_rows_gen, GEN_TEST_DATA[key], atol_diff2=1e-1)
        
    def test_convert_tohist(self):
        key = 'hist'
        self._run_convert(qp.hist_rows_gen, GEN_TEST_DATA[key], atol_diff=HIST_TOL)

    def test_convert_tohist_samples(self):
        key = 'hist_samples'
        self._run_convert(qp.hist_rows_gen, GEN_TEST_DATA[key], atol_diff=HIST_TOL)
        
    def test_convert_toquant(self):
        key = 'quant'
        self._run_convert(qp.quant_rows_gen, GEN_TEST_DATA[key])
        
    def test_convert_tomixmod(self):
        key = 'mixmod'
        self._run_convert(qp.mixmod_rows_gen, GEN_TEST_DATA[key])

    def test_convert_toflex(self):
        key = 'flex'
        self._run_convert(qp.flex_rows_gen, GEN_TEST_DATA[key])

        
        
if __name__ == '__main__':
    unittest.main()
