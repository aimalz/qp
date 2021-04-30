import numpy as np

from scipy.special import voigt_profile
from scipy.stats import rv_continuous
from scipy import integrate as sciint

from qp.pdf_gen import Pdf_rows_gen
from qp.factory import add_class
from qp.conversion_funcs import extract_voigt_mixmod, extract_voigt_xy, extract_voigt_xy_sparse


class voigt_gen(Pdf_rows_gen):
    """ Voigt mixture model based distribution

    Notes
    -----
    This implements a PDF using a Voigt mixture model,
    that generalizes the Gaussian mixture model implemented in mixmod_pdf.py
    """

    name = 'voigt'
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, means, stds, weights, gammas, *args, **kwargs):
        """
        create the Voigt mixture model using the input parameters

        Parameters
        ----------
        means : array_like
            The means of the Gaussians
        stds:  array_like
            The standard deviations of the Voigt function
        weights : array_like
            The weights to attach to the Voigt function
        gammas : array_like
            The extra parameters that generalize a Gaussian into a Voigt function
        """
        super(voigt_gen, self).__init__(*args, **kwargs)
        self._means = means
        self._stds = stds
        self._weights = weights
        self._gammas = gammas
        self._addobjdata('weights', self._weights)
        self._addobjdata('stds', self._stds)
        self._addobjdata('means', self._means)
        self._addobjdata('gammas', self._gammas)

    @property
    def weights(self):
        """Return weights to attach to the Gaussians"""
        return self._weights

    @property
    def means(self):
        """Return means of the Gaussians"""
        return self._means

    @property
    def stds(self):
        """Return standard deviations of the Gaussians"""
        return self._stds

    @property
    def gammas(self):
        """Return gamma of the Voigt"""
        return self._gammas

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        factored, xr, rr, _ = self._sliceargs(x, row)
        if factored:
            voigt_funcs = voigt_profile(np.expand_dims(xr, 0) - np.expand_dims(self._means[rr], -1), np.expand_dims(self._stds[rr], -1), np.expand_dims(self._gammas[rr], -1))
            #n = np.repeat(sla.norm(voigt_funcs, axis=2), xr.shape).reshape(voigt_funcs.shape)
            #voigt_funcs /= n
            #voigt_funcs *= np.expand_dims(self.weights[rr], -1)
            pdf = voigt_funcs.sum(axis=1)
            pdf /= sciint.trapz(pdf, xr)
            return pdf.reshape(xr.shape)

        voigt_funcs = voigt_profile(xr - self._means[rr].T, self._stds[rr].T, self._gammas[rr].T)
        voigt_funcs *= self.weights[rr].T
        pdf = voigt_funcs.sum()
        return pdf.reshape(x.shape)


    # def _voigt_cdf(self, x, mu, sigma, gamma):
    #     z = 1j * gamma
    #     z += x
    #     z /=(sqrt(2)*sigma)
    #     val = Real(0.5 + np.erf(z)/2 + 1j*z*z/np.pi * hyp2f2(1,1;1.5,2;-z*z) )
    # hyp2f2(1,1;1.5,2;-z*z) = hyp2f2(1,1;2,1.5;-z*z) = 1/(1+1-1.5)*hyp1f1(0.5;1.5;z)*hyp1f1(0.5;1.5;-z)
    #cf https://functions.wolfram.com/HypergeometricFunctions/Hypergeometric2F2/03/02/04/0002/

    def _updated_ctor_param(self):
        #pragma: no cover
        """
        Add gamma to the parameters
        """
        dct = super(voigt_gen, self)._updated_ctor_param()
        dct['means'] = self._means
        dct['stds'] = self._stds
        dct['weights'] = self._weights
        dct['gammas'] = self._gammas
        return dct

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        cls._add_extraction_method(extract_voigt_mixmod, None)
        cls._add_extraction_method(extract_voigt_xy, 'xy')
        cls._add_extraction_method(extract_voigt_xy_sparse, 'sparse')

voigt = voigt_gen.create

add_class(voigt_gen)
