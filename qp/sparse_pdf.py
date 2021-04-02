"""This module implements a PDT distribution sub-class using a Gaussian mixture model
"""

import numpy as np

from scipy.stats import rv_continuous
from scipy import stats as sps
from scipy import special

from qp.pdf_gen import Pdf_rows_gen
#from qp.conversion_funcs import extract_sparse_fit_values
#from qp.test_data import WEIGHT_MIXMOD, MEAN_MIXMOD, STD_MIXMOD, TEST_XVALS
from qp.factory import add_class
from qp.utils import reshape_to_pdf_size

from voigt_pdf import voigt_gen

def get_N(longN):
    """
    Extract coefficients fro the 32bits integer,
    Extract Ncoef and Nbase from 32 bit integer
    return (longN >> 16), longN & 0xffff

    :param int longN: input 32 bits integer

    :return: Ncoef, Nbase both 16 bits integer
    """
    return (longN >> 16), (longN & (2 ** 16 - 1))

def reconstruct_pdf_v(index, vals, zfine, mu, Nmu, sigma, Nsigma, Nv, cut=1.e-5):
    """
    This function reconstruct the pdf from the indices and values and parameters used to create the dictionary with
    Gaussians and Voigt profiles

    :param int index: List of indices in the dictionary for the selected bases
    :param float vals: values or coefficients corresponding to the listed indices
    :param float zfine: redshift values from the original pdf or used during the sparse representation
    :param float mu: [min_mu, max_mu] values used to create the dictionary
    :param int Nmu: Number of mu values used to create the dictionary
    :param float sigma: [min_sigma, mas_sigma] sigma values used to create the dictionary
    :param int Nsigma: Number of sigma values
    :param int Nv: Number of Voigt profiles used to create dictionary
    :param float cut: cut threshold when creating the dictionary

    :return: the pdf normalized so it sums to one
    """

    zmid = np.linspace(mu[0], mu[1], Nmu)
    sig = np.linspace(sigma[0], sigma[1], Nsigma)
    gamma = np.linspace(0, 0.5, Nv)
    pdf = np.zeros(len(zfine))
    for kk in range(len(index)):
        i = int(index[kk] / (Nsigma * Nv))
        j = int((index[kk] % (Nsigma * Nv)) / Nv)
        k = int(index[kk] % (Nsigma * Nv)) % Nv
        #print(i,j,k)
        pdft = special.voigt_profile(zfine-zmid[i], sig[j], sig[j] * gamma[k])
        pdft = np.where(pdft >= cut, pdft, 0.)
        pdft = pdft / np.linalg.norm(pdft)
        pdf += pdft * vals[kk]
        #pdf = where(pdf >= cut, pdf, 0)
    pdf = np.where(np.greater(pdf, np.max(pdf) * 0.005), pdf, 0.)
    if np.sum(pdf) > 0: pdf = pdf / np.sum(pdf)
    return pdf

def create_gaussian_dict(zfine, mu, Nmu, sigma, Nsigma, cut=1.e-5):
    """
    Creates a gaussian dictionary only

    :param float zfine: the x-axis for the PDF, the redshift resolution
    :param float mu: [min_mu, max_mu], range of mean for gaussian
    :param int Nmu: Number of values between min_mu and max_mu
    :param float sigma: [min_sigma, max_sigma], range of variance for gaussian
    :param int Nsigma: Number of values between min_sigma and max_sigma
    :param float cut: Lower cut for gaussians

    :return: Dictionary as numpy array with shape (len(zfine), Nmu*Nsigma)
    :rtype: float
    """

    zmid = np.linspace(mu[0], mu[1], Nmu)
    sig = np.linspace(sigma[0], sigma[1], Nsigma)
    NA = Nmu * Nsigma
    Npdf = len(zfine)
    A = np.zeros((Npdf, Nmu * Nsigma))
    k = 0
    for i in range(Nmu):
        for j in range(Nsigma):
            pdft = 1. * np.exp(-((zfine - zmid[i]) ** 2) / (2. * sig[j] * sig[j]))
            pdft = np.where(pdft >= cut, pdft, 0.)
            #pdft = np.where(greater(pdft, max(pdft) * 0.005), pdft, 0.)
            A[:, k] = pdft / np.linalg.norm(pdft)
            k += 1
    return A


class sparse_gen(voigt_gen):

    name = 'sparse'
    version = 0

    _support_mask = rv_continuous._support_mask

    def __init__(self, sparse_indices, metadata, *args, **kwargs):

        
        
        super(sparse_gen, self).__init__(*args, **kwargs)
        sparse_rep = reshape_to_pdf_size(sparse_indices, -1)
        self._metadata = metadata
        self._sparse_indices = sparse_indices
        self._addmetadata('metadata', metadata)
        self._addobjdata('pdfs', sparse_rep)

    def _pdf(self, x, row):
        # pylint: disable=arguments-differ
        header = self._metadata
        Ncoef = header['Ncoef']
        zfine = header['z']
        mu = header['mu']
        Nmu = header['Nmu']
        sigma = header['sig']
        Nsigma = header['Nsig']
        Nv = header['Nv']

        #JCT VALS = np.linspace(0, 1, Ncoef)
        #JCT dVals = VALS[1] - VALS[0]
        dVals = 1./(Ncoef-1)
        long_index = self._sparse_indices[row]
        sp_ind = np.array(list(map(get_N, long_index)))
        spi = sp_ind[:, 0]
        Dind2 = sp_ind[:, 1]
        vals = spi * dVals
        ####
        vals[0]=1.
        ####
        rep_pdf = reconstruct_pdf_v(Dind2, vals, x, mu, Nmu, sigma, Nsigma, Nv)
        return rep_pdf
        
        

    @classmethod
    def add_mappings(cls):
        """
        Add this classes mappings to the conversion dictionary
        """
        cls._add_creation_method(cls.create, None)
        #cls._add_extraction_method(extract_sparse_fit_values, None)


sparse = sparse_gen.create

# sparse_gen.test_data = dict(sparse=dict(gen_func=sparse,\
#                                                  ctor_data=dict(weights=WEIGHT_SPARSE,\
#                                                                     means=MEAN_SPARSE,\
#                                                                     stds=STD_SPARSE),\
#                                                  convert_data=dict(), test_xvals=TEST_XVALS,
#                                                  atol_diff2=1.))
add_class(sparse_gen)
