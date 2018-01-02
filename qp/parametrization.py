import numpy as np
import scipy.stats as sps

class Parametrization(object):
    """
    An object defining a general parametrization of a PDF
    """
    def __init__(self, format_type, metaparameters, parameters, name=None, vb=True):
        """
        Parameters
        ----------
        format_type: string
            name of the format
        metaparameters: numpy.ndarray, float
            the parameters that could be shared across an Ensemble object
        parameters: numpy.ndarray, float
            the parameters unique to the PDF object
        name: string, optional
            a name for the parametrization if multiple parametrizations of the same format are to be associated with a single PDF object
        vb: boolean, optional
            report on progress to stdout?

        Notes
        -----
        Includes default behavior for a generic format, currently error messages to be overridden by subclasses
        """
        self.format = format_type
        self.metaparameters = metaparameters
        self.parameters = parameters
        if name is None:
            self.name = self.format
        else:
            self.name = name

        self.pdf_evaluator = None
        self.cdf_evaluator = None
        self.ppf_evaluator = None
        self.rvs_evaluator = None
        self.fit_evaluator = None

    def _pdf(self, **kwargs):
        """
        Makes a function that evaluates the PDF based on the internal format
        """
        def pdf(x):# x points
            y = x
            print 'This is a dummy function.'
            return y
        self.pdf_evaluator = pdf
        return self.pdf_evaluator

    def _cdf(self, **kwargs):
        """
        Makes a function that evaluates the CDF based on the internal format
        """
        def cdf(x):# x points
            y = x
            print 'This is a dummy function.'
            return y
        self.cdf_evaluator = cdf
        return self.cdf_evaluator

    def _ppf(self, **kwargs):
        """
        Makes a function that evaluates the PPF based on the internal format
        """
        def ppf(x):# cdf points
            y = x
            print 'This is a dummy function.'
            return y
        self.ppf_evaluator = ppf
        return self.ppf_evaluator

    def _rvs(self, **kwargs):
        """
        Makes a function that draws samples based on the internal format
        """
        def rvs(x):# N
            y = x
            print 'This is a dummy function.'
            return y
        self.rvs_evaluator = rvs
        return  self.rvs_evaluator

    def _fit(self, **kwargs):
        """
        Returns a function that fits to a given functional form
        """
        def fit(x):# funcform
            y = x
            print 'This is a dummy function.'
            return y
        self.fit_evaluator = fit
        return self.fit_evaluator

    def convert(self, format_type, metaparameters, name=None, vb=True, **kwargs):
        """
        Converts to a different parametrization

        Parameters
        ----------
        format_type: string
            format of the target parametrization
        metaparameters: integer or numpy.ndarray, float or dict
            metaparameter(s) for the target parametrization
        name: string, optional
            name of the target parametrization
        vb: boolean, optional
            report on progress to stdout?
        kwargs: dict, optional
            optional keyword arguments for the conversion method

        Returns
        -------
        parametrization: qp.Parametrization object
            the target parametrization object
        """
        if format_type == 'funcform':
            if self.fit_evaluator is None:
                evaluator = self._fit(kwargs)
            else:
                evaluator = self.fit_evaluator
            parameters = evaluator(metaparameters)
            target = FuncForm(metaparameters, parameters, name=name, vb=vb)
        if format_type == 'pointeval':
            if self.pdf_evaluator is None:
                evaluator = self._pdf(kwargs)
            else:
                evaluator = self.pdf_evaluator
            parameters = evaluator(metaparameters)
            target = PointEval(metaparameters, parameters, name=name, vb=vb)
        if format_type == 'quantile':
            if self.ppf_evaluator is None:
                evaluator = self._ppf(kwargs)
            else:
                evaluator = self.ppf_evaluator
            parameters = evaluator(metaparameters)
            target = Quantile(metaparameters, parameters, name=name, vb=vb)
        if format_type == 'sample':
            if self.rvs_evaluator is None:
                evaluator = self._rvs(kwargs)
            else:
                evaluator = self.rvs_evaluator
            parameters = evaluator(metaparameters)
            target = Sample(metaparameters, parameters, name=name, vb=vb)
        return target# new_parametrization

class FuncForm(Parametrization):
    """
    A parametrization of a format with a functional form
    """
    def __init__(self, metaparameters, parameters, name=None, vb=True):
        """
        Parameters
        ----------
        metaparameters: numpy.ndarray, float
            the parameters that could be shared across an Ensemble object
        parameters: numpy.ndarray, float
            the parameters unique to the PDF object
        name: string, optional
            a name for the parametrization if multiple parametrizations of the same format are to be associated with a single PDF object
        vb: boolean, optional
            report on progress to stdout?
        """
        Parametrization.__init__(self, 'funcform', metaparameters, parameters, name=name, vb=vb)

    def _pdf(self, **kwargs):
        """
        Makes a function that evaluates the PDF of a functional form
        """
        def pdf(x):# x points
            y = self.metaparameters['rv_continuous'](loc=self.parameters['loc'], scale=self.parameters['scale'])
            return y.pdf(x)
        self.pdf_evaluator = pdf
        return self.pdf_evaluator

    def _cdf(self, **kwargs):
        """
        Makes a function that evaluates the CDF of a functional form
        """
        self.super()

    def _ppf(self, **kwargs):
        """
        Makes a function that evaluates the PPF of a functional form
        """
        self.super()

    def _rvs(self, **kwargs):
        """
        Makes a function that draws samples from a functional form
        """
        self.super()

    def _fit(self, **kwargs):
        """
        Returns a function that fits to a different functional form
        """
        self.super()

    def convert(self, format_type, metaparameters, name=None, vb=True, **kwargs):
        """
        Converts to a different parametrization

        Parameters
        ----------
        format_type: string
            format of the target parametrization
        metaparameters: integer or numpy.ndarray, float
            metaparameter(s) for the target parametrization
        name: string, optional
            name of the target parametrization
        vb: boolean, optional
            report on progress to stdout?
        kwargs: dict, optional
            optional keyword arguments for the conversion method

        Returns
        -------
        parametrization: qp.Parametrization object
            the target parametrization object
        """
        if format_type == 'funcform':
            if self.fit_evaluator is None:
                evaluator = self._fit(kwargs)
            else:
                evaluator = self.fit_evaluator
            parameters = evaluator(metaparameters)
            target = FuncForm(metaparameters, parameters, name=name, vb=vb)
        if format_type == 'pointeval':
            if self.pdf_evaluator is None:
                evaluator = self._pdf()
            else:
                evaluator = self.pdf_evaluator
            parameters = evaluator(metaparameters)
            target = PointEval(metaparameters, parameters, name=name, vb=vb)
        if format_type == 'quantile':
            if self.ppf_evaluator is None:
                evaluator = self._ppf(kwargs)
            else:
                evaluator = self.ppf_evaluator
            parameters = evaluator(metaparameters)
            target = Quantile(metaparameters, parameters, name=name, vb=vb)
        if format_type == 'sample':
            if self.rvs_evaluator is None:
                evaluator = self._rvs(kwargs)
            else:
                evaluator = self.rvs_evaluator
            parameters = evaluator(metaparameters)
            target = Sample(metaparameters, parameters, name=name, vb=vb)
        return target# new_parametrization

class PointEval(Parametrization):
    """
    A parametrization of the point evaluations format
    """
    def __init__(self, metaparameters, parameters, name=None, vb=True):
        Parametrization.__init__(self, 'pointeval', metaparameters, parameters, name=name, vb=vb)

class Quantile(Parametrization):
    """
    A parametrization of the quantile format
    """

class Sample(Parametrization):
    """
    A parametrization of the sample format
    """

# class StepFunc(Parametrization):
#     """
#     A parametrization of the step function (histogram) format
#     """

# class SparsePZ(Parametrization):
#     """
#     A parametrization of the SparsePZ format
#     """

# class FlexZBoost(Parametrization):
#     """
#     A parametrization of the FlexZBoost format
#     """
#
# . . .
