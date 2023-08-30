import unittest

import numpy as np

import qp
from qp.quantile_pdf_constructors import AbstractQuantilePdfConstructor, \
    DualSplineAverage


class DualSplineAverageTestCase(unittest.TestCase):
    """Tests for the CDF Spline Derivative PDF constructor for quantile parameterization."""

    def setUp(self):
        self.single_norm = qp.stats.norm(loc=3, scale=0.5)
        self.many_norm = qp.stats.norm(loc=np.array([[1], [2.5], [3]]), scale=np.array([[0.25], [0.5], [0.1]]))

        self.user_defined_quantiles = np.linspace(0.001, 0.999, 16)
        self.user_defined_grid = np.linspace(0, 5, 100)
        self.pdf_constructor = DualSplineAverage

    def test_instantiation(self):
        """Base case make sure that we can instantiate the class"""
        user_defined_locations = self.single_norm.ppf(self.user_defined_quantiles)
        pdf_constructor = self.pdf_constructor(quantiles=self.user_defined_quantiles, locations=user_defined_locations)
        assert isinstance(pdf_constructor, AbstractQuantilePdfConstructor)

    def test_instantiation_for_multiple_distributions(self):
        """Base case make sure that we can instantiate a pdf reconstructor for multiple distributions"""
        user_defined_locations = self.many_norm.ppf(self.user_defined_quantiles)
        pdf_constructor = self.pdf_constructor(quantiles=self.user_defined_quantiles, locations=user_defined_locations)
        assert isinstance(pdf_constructor, AbstractQuantilePdfConstructor)

    def test_debug(self):
        """Ensure that debug returns expected values before `prepare_constructor` has been run"""
        user_defined_locations = self.single_norm.ppf(self.user_defined_quantiles)
        pdf_constructor = self.pdf_constructor(quantiles=self.user_defined_quantiles, locations=user_defined_locations)
        debug_quantiles, debug_locations, debug_p_of_zs, debug_y1, debug_y2 = pdf_constructor.debug()

        print(debug_quantiles)
        assert np.all(np.isclose(debug_quantiles, self.user_defined_quantiles))
        assert np.all(np.isclose(debug_locations, user_defined_locations))
        self.assertIsNone(debug_p_of_zs)
        self.assertIsNone(debug_y1)
        self.assertIsNone(debug_y2)

    def test_basic_construct_pdf(self):
        """Base case to ensure that `construct_pdf` method runs with minimum arguments for single distribution case
        Want to verify only that the machinery is working, and the result is not just zeros."""
        user_defined_locations = self.single_norm.ppf(self.user_defined_quantiles)
        pdf_constructor = self.pdf_constructor(quantiles=self.user_defined_quantiles, locations=user_defined_locations)
        results = pdf_constructor.construct_pdf(self.user_defined_grid)
        self.assertIsNot(np.sum(results), 0.0)

    def test_basic_construct_pdf_for_multiple_distributions(self):
        """Base case to ensure that `construct_pdf` method runs with minimum arguments for many-distribution case
        Want to verify only that the machinery is working, and the result is not just zeros."""
        user_defined_locations = self.many_norm.ppf(self.user_defined_quantiles)
        pdf_constructor = self.pdf_constructor(quantiles=self.user_defined_quantiles, locations=user_defined_locations)
        results = pdf_constructor.construct_pdf(self.user_defined_grid)
        self.assertIsNot(np.sum(results), 0.0)
        self.assertEqual(len(results), 3)

    def test_basic_construct_pdf_for_subset_of_multiple_distributions(self):
        """Base case to ensure that `construct_pdf` method runs with minimum arguments for many-distribution case
        when passing in a `row` value. Want to verify only that the machinery is working, and that the size of the
        output matches expectations"""
        user_defined_locations = self.many_norm.ppf(self.user_defined_quantiles)
        pdf_constructor = self.pdf_constructor(quantiles=self.user_defined_quantiles, locations=user_defined_locations)
        user_defined_rows = [0,1]
        results = pdf_constructor.construct_pdf(self.user_defined_grid, row=user_defined_rows)
        self.assertEqual(len(results), 2)