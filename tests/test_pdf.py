"""
Unit tests for PDF class
"""
import sys
import os
import numpy as np, scipy.stats as sps
import unittest
import qp

from data import *


class PDFTestCase(unittest.TestCase):

    def setUp(self):
        """
        Make any objects that are used in multiple tests.
        """
        pass
        
    def tearDown(self):
        "Clean up any mock data files created by the tests."
        pass


    def _run_pdf_func_tests(self, key, test_data, short=False, check_props=True):
        """Run the test for a practicular class"""

        gen_func = test_data['gen_func']
        
        try:
            pdf = gen_func(**test_data['ctor_data'])
        except Exception as msg:
            raise ValueError("Failed to make %s %s %s" % (key, test_data['ctor_data'], msg))
        xpts = test_data['test_xvals']
        
        if check_props:
            # if we used the c'tor, make sure the class keeps the data used in the c'tor
            for key, val in test_data['ctor_data'].items():
                test_val = pdf.kwds.get(key, None)
                if test_val is None:
                    if not hasattr(pdf.dist, key):
                        raise ValueError("%s %s" % (pdf.dist, key))
            
        #FIXME
        if pdf.dist.npdf is not None:
            assert pdf.dist.npdf == pdf.npdf
        assert pdf.npdf == NPDF
        
        pdfs = pdf.pdf(xpts)

        if short:
            return pdf
        
        cdfs = pdf.cdf(xpts)
        quants = np.linspace(0.01, 0.99, 50)
        
        binw = xpts[1:] - xpts[0:-1]
        check_cdf = ((pdfs[:,0:-1] + pdfs[:,1:]) * binw /2).cumsum(axis=1) - cdfs[:,1:]
        assert_all_small(check_cdf, atol=1e-1)
    
        ppfs = pdf.ppf(quants)
        check_ppf = pdf.cdf(ppfs) - quants
        assert_all_small(check_ppf, atol=1e-2)

        sfs = pdf.sf(xpts)
        check_sf = sfs + cdfs
        assert_all_small(check_sf - 1, atol=1e-5)
        
        isfs = pdf.isf(quants)
        check_isf = pdf.cdf(ppfs) + quants[::-1]
        assert_all_small(check_isf - 1, atol=1e-2)
        return pdf

    def test_norm(self):
        key = 'norm'
        pdf = self._run_pdf_func_tests(key, GEN_TEST_DATA[key])

    def test_interp_fixed(self):
        key = 'interp_fixed'
        pdf = self._run_pdf_func_tests(key, GEN_TEST_DATA[key])

    def test_interp(self):
        key = 'interp'
        pdf = self._run_pdf_func_tests(key, GEN_TEST_DATA[key])
        
    def test_spline(self):
        key = 'spline'
        pdf = self._run_pdf_func_tests(key, GEN_TEST_DATA[key])

    def test_spline_xy(self):
        key = 'spline_xy'
        pdf = self._run_pdf_func_tests(key, GEN_TEST_DATA[key], check_props=False)

    def test_spline_kde(self):
        key = 'spline_kde'
        pdf = self._run_pdf_func_tests(key, GEN_TEST_DATA[key], check_props=False)

    def test_hist(self):
        key = 'hist'
        pdf = self._run_pdf_func_tests(key, GEN_TEST_DATA[key])

    def test_quant(self):
        key = 'quant'
        pdf = self._run_pdf_func_tests(key, GEN_TEST_DATA[key])

    def test_mixmod(self):
        key = 'mixmod'
        pdf = self._run_pdf_func_tests(key, GEN_TEST_DATA[key])

    def test_flex(self):
        key = 'flex'
        pdf = self._run_pdf_func_tests(key, GEN_TEST_DATA[key], short=True)


        
                                           

if __name__ == '__main__':
    unittest.main()
