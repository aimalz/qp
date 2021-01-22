"""
Unit tests for PDF class
"""
import numpy as np
import unittest
import qp

from qp import test_data

class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self.files = []
        
    def tearDown(self):
        "Clean up any mock data files created by the tests."
        for ff in self.files:
            os.unlink(ff)
        
    def test_profile(self):

        npdf = 100
        x_bins = test_data.XBINS
        x_cents = qp.utils.edge_to_center(x_bins)
        nbin = x_cents.size
        x_data = (np.ones((npdf, 1))*x_cents).T
        c_vals = np.linspace(0.5, 2.5, nbin)        
        y_data = np.expand_dims(c_vals, -1)*(0.95 + 0.1*np.random.uniform(size=(nbin, npdf)))
        pf_1 = qp.utils.profile(x_data.flatten(), y_data.flatten(), x_bins, std=False)
        pf_2 = qp.utils.profile(x_data.flatten(), y_data.flatten(), x_bins, std=True)
        qp.test_funcs.assert_all_close(pf_1[0], c_vals, atol=0.02, test_name="profile_mean")
        qp.test_funcs.assert_all_close(pf_1[0], pf_2[0], test_name="profile_check")
        qp.test_funcs.assert_all_close(pf_1[1], c_vals/npdf*np.sqrt(12), atol=0.2, test_name="profile_std")
        qp.test_funcs.assert_all_close(pf_1[1], 0.1*pf_2[1], test_name="profile_err")

        
if __name__ == '__main__':
    unittest.main()
