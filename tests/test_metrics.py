"""
Unit tests for PDF class
"""
import sys
import os
import numpy as np, scipy.stats as sps
import unittest
import qp

from qp import test_funcs


class MetricTestCase(unittest.TestCase):

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self.ens_n = test_funcs.build_ensemble(qp.stats.norm_gen.test_data['norm'])
        self.ens_n_shift = test_funcs.build_ensemble(qp.stats.norm_gen.test_data['norm_shifted'])
        
    def tearDown(self):
        "Clean up any mock data files created by the tests."
        pass
        
        
    def test_kld(self):
        kld = self.ens_n.kld(self.ens_n_shift, limits=(0,2.5))
        kld_check = qp.metrics.calculate_kld(self.ens_n, self.ens_n_shift, limits=(0.,2.5))
        assert np.allclose(kld, kld_check, atol=1e-2)
        
    def test_rmse(self):
        rmse = self.ens_n.rmse(self.ens_n_shift, limits=(0.,2.5))
        rmse_check = qp.metrics.calculate_rmse(self.ens_n, self.ens_n_shift, limits=(0.,2.5))
        assert np.allclose(rmse, rmse_check, atol=1e-2)

                                           

if __name__ == '__main__':
    unittest.main()
