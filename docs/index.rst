============================================
qp : quantile-parametrized PDF approximation
============================================

In a scientific inference we typically seek to characterize the
posterior probability density function (PDF) for our parameter(s),
which means we need to fund a suitable, calculable approximation to  it.
Popular choices include an ensemble of samples, a histogram estimator
based on those samples, or (in 1 dimensional problems) a tabulation of
the PDF on a regular parameter grid. `qp` is a python package that
supports these approximations, as well as the "quantile
parameterization" from which the package gets its name.

Tutorials
=========

See the following IPython Notebooks for some examples of using `qp`:

* `Basic Demo <http://htmlpreview.github.io/?https://github.com/aimalz/qp/blob/html/demo.html>`_
* `KL Divergence Illustration <http://htmlpreview.github.io/?https://github.com/aimalz/qp/blob/html/kld.html>`_


API Documentation
=================

`qp` provides a `PDF` class object, that builds on the
`scipy.stats` distributions to provide various approximate forms.
The package also contains some `utils` for quantifying the quality of
these approximations.


The PDF Class
-------------

.. autoclass:: pdf.PDF
    :members:

    .. automethod:: pdf.PDF.__init__


Quantification Utilities
------------------------

.. automodule:: utils
