"""
Unit tests for PDF class
"""
import sys
import os
import numpy as np, scipy.stats as sps
import unittest
import qp

from data import *


class MetricTestCase(unittest.TestCase):

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self.ens_n = build_ensemble(GEN_TEST_DATA['norm'])
        self.ens_n_shift = build_ensemble(GEN_TEST_DATA['norm_shifted'])
        
    def tearDown(self):
        "Clean up any mock data files created by the tests."
        pass
        
        
    def test_kld(self):
        kld = self.ens_n.kld(self.ens_n_shift, limits=(-1,1))
        kld_check = qp.calculate_kld(self.ens_n, self.ens_n_shift, limits=(-1,1))
        assert np.allclose(kld, kld_check, atol=1e-2)
        
    def test_rmse(self):
        rmse = self.ens_n.rmse(self.ens_n_shift, limits=(-1,1))
        rmse_check = qp.calculate_rmse(self.ens_n, self.ens_n_shift, limits=(-1,1))
        assert np.allclose(rmse, rmse_check, atol=1e-2)

                                           

if __name__ == '__main__':
    unittest.main()
