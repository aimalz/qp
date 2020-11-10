"""This module implements continous distributions generator that uses the `flexcode`


Open questions:
1) At this time the normalization is not enforced for many of the PDF types.  It is assumed that
the user values give correct normalization.  We should think about this more.

2) At this time for most of the distributions, only the _pdf function is overridden.  This is all that
is required to inherit from `scipy.stats.rv_continuous`; however, providing implementations of some of
_logpdf, _cdf, _logcdf, _ppf, _rvs, _isf, _sf, _logsf could speed the code up a lot is some cases.

"""

import numpy as np

from scipy.stats import rv_continuous
from scipy.optimize import least_squares
from scipy.linalg import solve_triangular, cho_solve
from scipy.linalg import norm as sla_norm

from flexcode.basis_functions import evaluate_basis
from flexcode.helpers import box_transform

from .pdf_gen import Pdf_rows_gen
from .ensemble import Ensemble
from .persistence import register_pdf_class
from .conversion import register_class_conversions


def fit_flex_basis_funcs(basis, values):
    """Convert to a mixture model using a set of values sample from the pdf

    Parameters
    ----------
    basis : array_like
        Array with all basis on each column, must has shape (len(vector), total basis) and each column must have euclidean l-2 norm equal to 1
    values : array_like
        vector of which a sparse representation is desired

    Returns
    -------
    coefs : array_like
        Coefficients
    """
    def objective_func(coefs, row_vals):
        estim = np.sum(coefs*basis, axis=1)
        return np.square(row_vals - estim)

    def fit_row(row_vals):
        fit_result = least_squares(objective_func, np.zeros((basis.shape[-1])), kwargs=dict(row_vals=row_vals))
        return fit_result['x']

    vv = np.vectorize(fit_row, signature="(%i)->(%i)" % (values.shape[0], basis.shape[-1]))
    return vv(values)


def decompose_to_sparse_basis(basis, values, n_basis, tolerance=None):
    """Decompose a function into the basis using Cholesky decomposition

    Note
    ----
    This is copied form pdf_storage.sparse_basis by Matias Carrasco Kind
    If SparsePz is updated we should just use the version of the code from tehre

    Parameters
    ----------
    basis : array_like
        Array with all basis on each column, must has shape (len(vector), total basis) and each column must have euclidean l-2 norm equal to 1
    values : array_like
        vector of which a sparse representation is desired
    n_basis : int
        number of desired basis
    tolerance: float
        tolerance desired if n_basis is not needed to be fixed, must input a large number for n_basis to assure achieving tolerance

    Returns
    -------
    coefs : array_like
        Coefficients
    """
    a_n = np.zeros(basis.shape[1])
    machine_eps = np.finfo(values.dtype).eps
    alpha = np.dot(basis.T, values)
    res = values
    idxs = np.arange(basis.shape[1])  # keeping track of swapping
    L = np.zeros((n_basis, n_basis), dtype=values.dtype)
    L[0, 0] = 1.
    gamma_full = np.zeros((n_basis), dtype=values.dtype)

    for n_active in range(n_basis):
        lam = np.argmax(np.abs(np.dot(basis.T, res)))
        #if lam < n_active or alpha[lam] ** 2 < machine_eps:
        if alpha[lam] ** 2 < machine_eps:
            n_active -= 1
            break
        if n_active > 0:
            # Updates the Cholesky decomposition of basis
            L[n_active, :n_active] = np.dot(basis[:, :n_active].T, basis[:, lam])
            solve_triangular(L[:n_active, :n_active], L[n_active, :n_active], lower=True, overwrite_b=True)
            v = sla_norm(L[n_active, :n_active]) ** 2
            if 1 - v <= machine_eps:
                print("Selected basis are dependent or normed are not unity %.2f" % (1-v))
                break
            L[n_active, n_active] = np.sqrt(1 - v)
        basis[:, [n_active, lam]] = basis[:, [lam, n_active]]
        alpha[[n_active, lam]] = alpha[[lam, n_active]]
        idxs[[n_active, lam]] = idxs[[lam, n_active]]
        # solves LL'x = query_vec as a composition of two triangular systems
        gamma = cho_solve((L[:n_active + 1, :n_active + 1], True), alpha[:n_active + 1], overwrite_b=False)
        res = values - np.dot(basis[:, :n_active + 1], gamma)
        gamma_full[:n_active + 1] = gamma
        if tolerance is not None and sla_norm(res) ** 2 <= tolerance:
            break
    a_n[idxs[:n_active + 1]] = gamma
    return a_n
    ##return idxs[:n_active + 1], a_n
    #return idxs, gamma_full




def decompose_flex_basis_funcs(basis, values, nbasis, **kwargs):
    """Convert to a mixture model using a set of values sample from the pdf

    Parameters
    ----------
    basis : array_like
        Array with all basis on each column, must has shape (len(vector), total basis) and each column must have euclidean l-2 norm equal to 1
    values : array_like
        Values we are trying to decopmose
    n_basis : int
        number of desired basis

    Returns
    -------
    coefs : array_like
        Coefficients
    """
    tolerance = kwargs.pop('tolerance', None)

    def decompose_row(row_vals):
        result = decompose_to_sparse_basis(basis, row_vals, n_basis=nbasis, tolerance=tolerance)
        return result

    vv = np.vectorize(decompose_row, signature="(%i)->(%i)" % (values.shape[-1], basis.shape[-1]))
    vals = vv(values)
    return vals



def convert_to_flex(in_dist, class_to, **kwargs):
    """Convert to a mixture model using a set of values sample from the pdf

    Parameters
    ----------
    in_dist : `qp.Ensemble`
        Input ensemble
    class_to : `class`
        Class to convert to

    Keywords
    --------
    ncomps : `int`
        Number of components in mixture model to use
    nsamples : `int`
        Number of samples to generate
    Remaining keywords are passed to class constructor.

    Returns
    -------
    dist : An distrubtion object of type class_to, instantiated by fitting to the samples.
    """
    grid = kwargs.pop('grid', None)
    xmin = kwargs.pop('xmin', grid[0])
    xmax = kwargs.pop('xmax', grid[-1])
    nbasis = kwargs.pop('nbasis', 32)
    basis_system = kwargs.pop('basis_system', None)

    if grid is None: #pragma : no cover
        raise ValueError("You need to provide a grid to convert to flex basis")
    if basis_system is None: #pragma : no cover
        raise ValueError("You need to provide a basis_system to convert to flex basis")
    pdf_vals = in_dist.pdf(grid)

    basis = evaluate_basis(box_transform(np.expand_dims(grid, -1), xmin, xmax), nbasis, basis_system)
    #basis /= sla_norm(basis)
    #coefs = fit_flex_basis_funcs(basis, pdf_vals)

    norm = sla_norm(basis)
    basis /= norm
    pdf_vals /= norm
    coefs = decompose_flex_basis_funcs(basis, pdf_vals, nbasis=nbasis)
    return Ensemble(class_to, data=dict(coefs=coefs, basis_system=basis_system, z_min=xmin, z_max=xmax))


class flex_rows_gen(Pdf_rows_gen):
    """Flexcode based distribution

    Notes
    -----
    This implements a PDF using `flexcode`.

    """
    # pylint: disable=protected-access

    name = 'flex_dist'
    version = 0

    _support_mask = rv_continuous._support_mask


    def __init__(self, coefs, basis_system, z_min, z_max, *args, **kwargs):
        """
        Create a new distribution using the given histogram
        Parameters
        ----------
        coefs : array_like
          The basis_function coefficients
        """
        self._coefs = np.asarray(coefs)
        self._basis_system = basis_system

        # Set support
        kwargs['a'] = self.a = z_min
        kwargs['b'] = self.b = z_max
        kwargs['npdf'] = coefs.shape[0]

        super(flex_rows_gen, self).__init__(*args, **kwargs)
        self._addmetadata('basis_system', self._basis_system)
        self._addmetadata('z_min', self.a)
        self._addmetadata('z_max', self.b)
        self._addobjdata('coefs', self._coefs)


    @property
    def z_min(self):
        """Return the min of the basis range"""
        return self.a

    @property
    def z_max(self):
        """Return the max of the basis range"""
        return self.b

    @property
    def coefs(self):
        """Return the histogram bin edges"""
        return self._coefs

    @property
    def basis_system(self):
        """Return the histogram bin values"""
        return self._basis_system

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            x_trans = box_transform(xr, self.a, self.b)
            basis = evaluate_basis(np.expand_dims(x_trans, -1), self.coefs.shape[1], self.basis_system)
            return np.matmul(self._coefs[rr], basis.T).flatten()
        x_trans = box_transform(xr, self.a, self.b)
        basis = evaluate_basis(x_trans, self.coefs.shape[1], self.basis_system)
        return np.sum(self._coefs[rr]*basis, axis=1).flatten()

    def _updated_ctor_param(self):
        """
        Set the bins as additional constructor argument
        """
        dct = super(flex_rows_gen, self)._updated_ctor_param()
        dct['coefs'] = self._coefs
        dct['basis_system'] = self._basis_system
        dct['z_min'] = self.a
        dct['z_max'] = self.b

        return dct

    @classmethod
    def add_conversion_mappings(cls, conv_dict):
        """
        Add this classes mappings to the conversion dictionary
        """
        conv_dict.add_mapping((cls.create, convert_to_flex), cls, None, None)


flex = flex_rows_gen.create

register_class_conversions(flex_rows_gen)
register_pdf_class(flex_rows_gen)
