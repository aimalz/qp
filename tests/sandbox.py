

from data import *


xvals = GEN_TEST_DATA['flex']['test_xvals']
ens_f = build_ensemble(GEN_TEST_DATA['flex'])

pdfs = ens_f.pdf([1])
pdfs = ens_f.pdf(xvals)
cdfs = ens_f.cdf(xvals)
binw = xvals[1:] - xvals[0:-1]

check_cdf = ((pdfs[:,0:-1] + pdfs[:,1:]) * binw /2).cumsum(axis=1) - cdfs[:,1:]
