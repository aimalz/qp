"""
Unit tests for PDF class
"""
import sys
import os
import numpy as np, scipy.stats as sps
import unittest
import qp

from data import *


class PlottingTestCase(unittest.TestCase):

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        pass
        
    def tearDown(self):
        "Clean up any mock data files created by the tests."
        pass


    def _run_plotting_func_tests(self, test_data):
        """Run the test for a practicular class"""
        ens = build_ensemble(test_data)
        pdf = ens[0]
        fig, axes = qp.qp_plot_native(pdf, xlim=(-5, 5))
        assert fig is not None
        assert axes is not None
        fig, axes = qp.qp_plot(pdf, axes=axes)
        assert fig is not None
        assert axes is not None
        
        
    def test_norm(self):
        key = 'norm'
        self._run_plotting_func_tests(GEN_TEST_DATA[key])

    def test_interp(self):
        key = 'interp'
        self._run_plotting_func_tests(GEN_TEST_DATA[key])
        
    def test_spline(self):
        key = 'spline'
        self._run_plotting_func_tests(GEN_TEST_DATA[key])

    def test_hist(self):
        key = 'hist'
        self._run_plotting_func_tests(GEN_TEST_DATA[key])

    def test_quant(self):
        key = 'quant'
        self._run_plotting_func_tests(GEN_TEST_DATA[key])

    def test_kde(self):
        key = 'kde'
        self._run_plotting_func_tests(GEN_TEST_DATA[key])

    def test_mixmod(self):
        key = 'mixmod'
        self._run_plotting_func_tests(GEN_TEST_DATA[key])

        
                                           

if __name__ == '__main__':
    unittest.main()
