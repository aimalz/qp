`qp`
=======

In a scientific inference we typically seek to characterize the
posterior probability density function (PDF) for our parameter(s),
which means we need to fund a suitable, calculable approximation to  it.
Popular choices include an ensemble of samples, a histogram estimator
based on those samples, or (in 1 dimensional problems) a tabulation of
the PDF on a regular parameter grid. `qp` is a python package that
supports these approximations, as well as the "quantile
parameterization" from which the package gets its name. 


API
---

.. automodule:: qp.pdf

.. automodule:: qp.utils
