"""
Unit tests for PDF class
"""

import unittest
import qp
import numpy as np

from qp import test_funcs
from qp.utils import epsilon


class MetricTestCase(unittest.TestCase):
    """ Tests for the metrics """

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self.ens_n = test_funcs.build_ensemble(qp.stats.norm_gen.test_data['norm'])  #pylint: disable=no-member
        self.ens_n_shift = test_funcs.build_ensemble(qp.stats.norm_gen.test_data['norm_shifted'])  #pylint: disable=no-member

        locs = 2* (np.random.uniform(size=(10,1))-0.5)
        scales = 1 + 0.2*(np.random.uniform(size=(10,1))-0.5)
        self.ens_n_plus_one = qp.Ensemble(qp.stats.norm, data=dict(loc=locs, scale=scales)) #pylint: disable=no-member

        bins = np.linspace(-5, 5, 11)
        self.ens_s = self.ens_n.convert_to(qp.spline_gen, xvals=bins, method="xy")

    def tearDown(self):
        """ Clean up any mock data files created by the tests. """

    def test_kld(self):
        """ Test the calculate_kld method """
        kld = qp.metrics.calculate_kld(self.ens_n, self.ens_n_shift, limits=(0.,2.5))
        assert np.all(kld == 0.)

    def test_kld_different_shapes(self):
        """ Ensure that the kld function fails when trying to compare ensembles of different sizes. """
        with self.assertRaises(ValueError) as context:
            qp.metrics.calculate_kld(self.ens_n, self.ens_n_plus_one, limits=(0.,2.5))

        self.assertTrue('Cannot calculate KLD between two ensembles with different shapes' in str(context.exception))

    def test_broken_kld(self):
        """ Test to see if the calculate_kld function responds appropriately to negative results """
        kld = qp.metrics.calculate_kld(self.ens_n, self.ens_s, limits=(0.,2.5))

        assert np.all(kld == epsilon)
        

    def test_rmse(self):
        """ Test the calculate_rmse method """
        rmse = qp.metrics.calculate_rmse(self.ens_n, self.ens_n_shift, limits=(0.,2.5))
        assert np.all(rmse == 0.)

    def test_rmse_different_shapes(self):
        """ Ensure that the rmse function fails when trying to compare ensembles of different sizes. """
        with self.assertRaises(ValueError) as context:
            qp.metrics.calculate_rmse(self.ens_n, self.ens_n_plus_one, limits=(0.,2.5))

        self.assertTrue('Cannot calculate RMSE between two ensembles with different shapes' in str(context.exception))


if __name__ == '__main__':
    unittest.main()
