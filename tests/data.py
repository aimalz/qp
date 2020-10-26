

import numpy as np, scipy.stats as sps

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

HIST_TOL = 4./NBIN
FLEX_BASIS_SYST = 'cosine'
FLEX_COEFS = np.load('test_flex_coefs.npy', allow_pickle=True)


GEN_TEST_DATA = dict(
    norm=dict(gen_class=qp.norm_gen, ctor_data=dict(loc=LOC, scale=SCALE),\
                  convert_data=dict(), test_xvals=TEST_XVALS),
    norm_shifted=dict(gen_class=qp.norm_gen, ctor_data=dict(loc=LOC, scale=SCALE),\
                          convert_data=dict(), test_xvals=TEST_XVALS),
    interp=dict(gen_class=qp.interp_rows_gen, ctor_data=dict(xvals=XARRAY, yvals=YARRAY),\
                    convert_data=dict(xvals=XBINS), test_xvals=TEST_XVALS),
    spline=dict(gen_class=qp.spline_rows_gen, ctor_data=dict(xvals=XARRAY, yvals=YARRAY),\
                    convert_data=dict(xvals=XBINS), test_xvals=TEST_XVALS[::10]),
    hist=dict(gen_class=qp.hist_rows_gen, ctor_data=dict(bins=XBINS, pdfs=HIST_DATA),\
                  convert_data=dict(bins=XBINS), test_xvals=TEST_XVALS),
    quant=dict(gen_class=qp.quant_rows_gen, ctor_data=dict(quants=QUANTS, locs=QLOCS),\
                   convert_data=dict(quants=QUANTS), test_xvals=TEST_XVALS),
    kde=dict(gen_class=qp.kde_rows_gen, ctor_data=dict(samples=SAMPLES, xvals=np.linspace(-5, 5, 51), yvals=None),\
                   convert_data=dict(xvals=np.linspace(-5, 5, 51)), test_xvals=TEST_XVALS),\
    mixmod=dict(gen_class=qp.mixmod_rows_gen, ctor_data=dict(weights=WEIGHT_MIXMOD, means=MEAN_MIXMOD, stds=STD_MIXMOD),\
                    convert_data=dict(), test_xvals=TEST_XVALS),\
    flex=dict(gen_class=qp.flex_rows_gen, ctor_data=dict(coefs=FLEX_COEFS, basis_system=FLEX_BASIS_SYST, z_min=-5, z_max=5), 
                   convert_data=dict(grid=TEST_XVALS, basis_system=FLEX_BASIS_SYST), test_xvals=TEST_XVALS))


def assert_all_small(arr, atol):
    if not np.allclose(arr, 0, atol=atol):
        raise ValueError("%.2e %.2e %.2e" % (arr.min(), arr.max(), atol))


def build_ensemble(test_data):
    gen_class = test_data['gen_class']
    ctor_data = test_data['ctor_data']
    try:
        return qp.Ensemble(gen_class, data=ctor_data)
    except Exception as msg:
        raise ValueError("Failed to make %s %s %s" % (gen_class, ctor_data, msg))
  
