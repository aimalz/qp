"""
Unit tests for Formatter subclasses
"""
import os
import numpy as np
import scipy.stats as sps
import unittest

import qp
from qp.formats import *
import formatter.Formatter as Formatter
from qp import parametrization.Parametrization as Parametrization

class FormatTestCase(unittest.TestCase):

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        pass

    def tearDown(self):
        "Clean up any mock data files created by the tests."
        pass

    def test_registry_graph(self):
        """
        Make sure all formats in registry have a path to all others
        """
        for each_format in Formatter.registry:
            initial = Parametrization(each_format, None, None)
            for other_format in Formatter.registry:
                final = initial.find_shortest_path(other_format)


if __name__ == '__main__':
    unittest.main()
