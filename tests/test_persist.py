"""
Unit tests for PDF class
"""
import sys
import os
import numpy as np, scipy.stats as sps
import unittest
import qp

from data import *


class PersistTestCase(unittest.TestCase):

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        pass
        
    def tearDown(self):
        "Clean up any mock data files created by the tests."
        pass

    def _run_persist_func_tests(self, test_data):
        """Run the test for a practicular class"""
        ens = build_ensemble(test_data)
        ftypes = ['fits', 'hdf5']
        for ftype in ftypes:
            filename = "test_%s.%s" % ( ens.gen_class.name, ftype )
            ens.write_to(filename)
            ens_r = qp.Ensemble.read_from(filename)
            diff = ens.pdf(test_data['test_xvals']) - ens_r.pdf(test_data['test_xvals'])
            assert np.allclose(diff, 0, atol=1e-5)        
            os.unlink(filename)
            os.unlink(filename.replace(".%s" % ftype, "_meta.%s" % ftype))
            
    def test_norm(self):
        key = 'norm'
        self._run_persist_func_tests(GEN_TEST_DATA[key])

    def test_interp(self):
        key = 'interp'
        self._run_persist_func_tests(GEN_TEST_DATA[key])
        
    def test_spline(self):
        key = 'spline'
        self._run_persist_func_tests(GEN_TEST_DATA[key])

    def test_hist(self):
        key = 'hist'
        self._run_persist_func_tests(GEN_TEST_DATA[key])

    def test_quant(self):
        key = 'quant'
        self._run_persist_func_tests(GEN_TEST_DATA[key])

    def test_mixmod(self):
        key = 'mixmod'
        self._run_persist_func_tests(GEN_TEST_DATA[key])

    def test_flex(self):
        key = 'flex'
        self._run_persist_func_tests(GEN_TEST_DATA[key])

if __name__ == '__main__':
    unittest.main()
