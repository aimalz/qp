"""
Unit tests for PDF class
"""
import sys
import os
import numpy as np, scipy.stats as sps
import unittest
import qp

from qp import test_data


class InfrastructureTestCase(unittest.TestCase):

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        self.files = []
        
    def tearDown(self):
        "Clean up any mock data files created by the tests."
        for ff in self.files:
            os.unlink(ff)

        
    def test_print_factory(self):
        qp.instance().pretty_print()

    def test_slice_dict(self):
        orig_dict = dict(loc=test_data.LOC, scale=test_data.SCALE, scalar=1)
        sliced = qp.dict_utils.slice_dict(orig_dict, 1)
        assert sliced['loc'] == test_data.LOC[1]
        assert sliced['scale'] == test_data.SCALE[1]
        assert sliced['scalar'] == 1
        
    def test_print_dict_shape(self):
        test_dict = dict(loc=test_data.LOC, scale=test_data.SCALE)
        qp.dict_utils.print_dict_shape(test_dict)

    def test_get_val_or_default(self):
        test_dict = dict(key=1)
        test_dict[None] = 2
        assert qp.dict_utils.get_val_or_default(test_dict, 'key') == 1
        assert qp.dict_utils.get_val_or_default(test_dict, 'nokey') == 2
        assert qp.dict_utils.get_val_or_default(test_dict, None) == 2
        assert qp.dict_utils.set_val_or_default(test_dict, 'key', 5) == 1

        test_dict.pop(None)
        assert qp.dict_utils.get_val_or_default(test_dict, 'nokey') == None

        
if __name__ == '__main__':
    unittest.main()
