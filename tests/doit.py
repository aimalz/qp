import numpy as np
import qp

NPDF = 11
NBIN = 21
NSAMPLES = 100
LOC = np.expand_dims(np.linspace(-1, 1, NPDF), -1)
SCALE = np.expand_dims(np.linspace(0.8, 1.2, NPDF), -1)
LOC_SHIFTED = LOC + SCALE
TEST_XVALS = np.linspace(-5, 5, 201)
XBINS = np.linspace(-5, 5, NBIN)
XARRAY = np.ones((NPDF, NBIN))*XBINS
YARRAY = np.expand_dims(np.linspace(0.8, 1.2, NPDF), -1)*(1. + 0.1*np.random.uniform(size=(NPDF, NBIN)))
HIST_DATA = YARRAY[:,0:-1]
QUANTS = np.linspace(0.01, 0.99, NBIN)
QLOCS = qp.norm(loc=LOC, scale=SCALE).ppf(QUANTS)
SAMPLES = qp.norm(loc=LOC, scale=SCALE).rvs(size=(NPDF, NSAMPLES))

MEAN_MIXMOD = np.vstack([np.linspace(-1, 1, NPDF), np.linspace(0, 1, NPDF), np.linspace(-1, 0, NPDF)]).T
STD_MIXMOD = np.vstack([np.linspace(0.8, 1.2, NPDF), np.linspace(0.8, 1.2, NPDF), np.linspace(0.8, 1.2, NPDF)]).T
WEIGHT_MIXMOD = np.vstack([0.7*np.ones((NPDF)), 0.2*np.ones((NPDF)), 0.1*np.ones((NPDF))]).T



pdf = qp.mixmod_rows_gen.create(weights=WEIGHT_MIXMOD, means=WEIGHT_MIXMOD, stds=STD_MIXMOD)
pdfs = pdf.pdf(TEST_XVALS)
cdfs = pdf.cdf(TEST_XVALS)
binw = TEST_XVALS[1:] - TEST_XVALS[0:-1]
check_cdf = ((pdfs[:,0:-1] + pdfs[:,1:]) * binw /2).cumsum(axis=1) - cdfs[:,1:]
