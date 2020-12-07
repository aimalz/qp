"""Module to define qp distributions that inherit from scipy distributions"""

from qp.test_data import LOC, SCALE, TEST_XVALS
from qp.factory import stats

# pylint: disable=no-member
stats.norm_gen.test_data = dict(norm=dict(gen_func=stats.norm, ctor_data=dict(loc=LOC, scale=SCALE),\
                                            test_xvals=TEST_XVALS, do_samples=True),
                            norm_shifted=dict(gen_func=stats.norm, ctor_data=dict(loc=LOC, scale=SCALE),\
                                                  test_xvals=TEST_XVALS))
