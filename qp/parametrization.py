import numpy as np
import scipy.interpolate as spi
import scipy.optimize as spo
import scipy.stats as sps

class Parametrization(object):
    """
    An object defining a general parametrization of a PDF
    """
    def __init__(self, format_type, metaparam, param, name=None, vb=True):
        """
        Parameters
        ----------
        format_type: string
            name of the format
        metaparam: numpy.ndarray, float
            the parameters that could be shared across an Ensemble object
        param: numpy.ndarray, float
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
        self.metaparam = metaparam
        self.param = param
        if name is None:
            self.name = self.format
        else:
            self.name = name

    def convert(self, format_type, metaparam, name=None, vb=True, **kwargs):
        """
        Converts to a different parametrization

        Parameters
        ----------
        format_type: string
            format of the target parametrization
        metaparam: integer or numpy.ndarray, float or dict
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
            param = evaluator(metaparam)
            target = FuncForm(metaparam, param, name=name, vb=vb)
        if format_type == 'pointeval':
            if self.pdf_evaluator is None:
                evaluator = self._pdf(kwargs)
            else:
                evaluator = self.pdf_evaluator
            param = evaluator(metaparam)
            target = PointEval(metaparam, param, name=name, vb=vb)
        if format_type == 'quantile':
            if self.ppf_evaluator is None:
                evaluator = self._ppf(kwargs)
            else:
                evaluator = self.ppf_evaluator
            param = evaluator(metaparam)
            target = Quantile(metaparam, param, name=name, vb=vb)
        if format_type == 'sample':
            if self.rvs_evaluator is None:
                evaluator = self._rvs(kwargs)
            else:
                evaluator = self.rvs_evaluator
            param = evaluator(metaparam)
            target = Sample(metaparam, param, name=name, vb=vb)
        return target# new_parametrization

class FuncForm(Parametrization):
    """
    A parametrization of a format with a functional form
    """
    def __init__(self, metaparam, param, name=None, vb=False):
        """
        Parameters
        ----------
        metaparam: dict
            the function and any parameters that would not be unique to the probability distribution
        param: dict
            the parameters unique to the probability distribution
        name: string, optional
            a name for the parametrization if multiple parametrizations of the same format are to be associated with a single PDF object
        vb: boolean, optional
            report on progress to stdout?

        Notes
        -----

        """
        Parametrization.__init__(self, 'funcform', metaparam, param, name=name, vb=vb)

        self.function = self.metaparam['function']
        params = self.metaparam
        params.update(self.param)
        params.pop('function')
        self.params = params
        self.representation = self.function(**self.params)

    def _pdf(self):
        """
        Makes a function that evaluates the PDF of a functional form
        """
        def pdf(x):# x points
            return self.representation.pdf(x)
        self.pdf_evaluator = pdf
        return self.pdf_evaluator

    def _cdf(self):
        """
        Makes a function that evaluates the CDF of a functional form
        """
        def cdf(x):# x points
            return self.representation.cdf(x)
        self.cdf_evaluator = cdf
        return self.cdf_evaluator

    def _ppf(self):
        """
        Makes a function that evaluates the PPF of a functional form
        """
        def ppf(x):# x points
            return self.representation.ppf(x)
        self.ppf_evaluator = ppf
        return self.ppf_evaluator

    def _rvs(self):
        """
        Makes a function that draws samples from a functional form
        """
        def rvs(x):# x points
            return self.representation.rvs(x)
        self.rvs_evaluator = rvs
        return self.rvs_evaluator

    def _fit(self):
        """
        Returns a function that fits to a different functional form
        """
        def fit(x):# x points
            return self.representation.fit(x)
        self.fit_evaluator = fit
        return self.fit_evaluator

    def convert(self, format_type, metaparam, name=None, vb=True, via=None):
        """
        Converts to a different parametrization

        Parameters
        ----------
        format_type: string
            format of the target parametrization
        metaparam: integer or numpy.ndarray, float
            metaparameter(s) for the target parametrization
        name: string, optional
            name of the target parametrization
        vb: boolean, optional
            report progress to stdout?
        via: dict, optional
            keywords for intermediate format

        Returns
        -------
        parametrization: qp.Parametrization object
            the target parametrization object

        Notes
        -----
        TO DO: Add in keywords in via for complicated conversion options
        """
        if name is None:
            name = self.name

        if format_type == 'funcform':
            if via is None:
                via = {'format_type': 'sample', 'metaparam': 100}
            if vb:
                print 'Converting to '+format_type+' via '+via['format_type']
            intermediate = self.convert(via['format_type'], via['metaparam'], vb=vb)
            target = intermediate.convert('funcform', metaparam, name=name, vb=vb)
        elif format_type == 'pointeval':
            try:
                evaluator = self.pdf_evaluator
            except AttributeError:
                evaluator = self._pdf()
            param = evaluator(metaparam)
            target = PointEval(metaparam, param, name=name, vb=vb)
        elif format_type == 'quantile':
            try:
                evaluator = self.ppf_evaluator
            except AttributeError:
                evaluator = self._ppf()
            param = evaluator(metaparam)
            target = Quantile(metaparam, param, name=name, vb=vb)
        elif format_type == 'sample':
            try:
                evaluator = self.rvs_evaluator
            except AttributeError:
                evaluator = self._rvs()
            param = evaluator(metaparam)
            target = Sample(param, name=name, vb=vb)
        else:
            print 'Conversion to ' + format_type + ' not supported'
            return

        return target# new_parametrization

class PointEval(Parametrization):
    """
    A parametrization of the point evaluations format
    """
    def __init__(self, metaparam, param, name=None, vb=True):
        """
        Parameters
        ----------
        metaparam: numpy.ndarray, float
            the independent variable values at which the probability distribution is evaluated
        param: numpy.ndarray, float
            the dependent variable values of the probability distribution
        name: string, optional
            a name for the parametrization if multiple parametrizations of the same format are to be associated with a single PDF object
        vb: boolean, optional
            report on progress to stdout?
        """
        Parametrization.__init__(self, 'pointeval', metaparam, param, name=name, vb=vb)

    def _curve_fit(self):
        """
        Returns a function that fits a functional form via optimization
        """
        def fit(function, metaparam, xdata, ydata, fit_args=None):
            def fun(x, param):
                y = qp.Parametrization.FuncForm(function, metaparam, param).pdf(x)
                return y
            popt, pcov = spo.curve_fit(fun, xdata, ydata, **fit_args)
            return popt
        self.curve_fit_evaluator = fit
        return self.curve_fit_evaluator

    def _interp1d(self):
        """
        Returns an interpolating function using `scipy.interpolate.interp1d`
        """
        def fit(x, xdata, ydata, fit_args=None):
            fun = spi.interp1d(xdata, ydata, **fit_args)
            return fun(x)
        self.interp1d_evaluator = fit
        return self.interp1d_evaluator

    def _spline(self):
        """
        Returns an interpolating function using `scipy.interpolate.InterpolatingUnivariateSpline`
        """
        def fun(xdata, ydata, fit_args=None):
            fun = spi.InterpolatedUnivariateSpline(xdata, ydata, **fit_args)
            return fun
        self.spline_evaluator = fun

        def antiderivative(x, fit_args=None):
            newfun = self.spline_evaluator.antiderivative(**fit_args)
            return newfun
        self.spline_antiderivative = antiderivative

        def derivative(x, fit_args=None):
            newfun = self.spline_evaluator.derivative(**fit_args)
            return newfun
            self.spline_derivative = derivative

        def integral(a, b):
            return self.spline_evaluator.integral(a, b)
        self.spline_integral = integral

        return self.spline_evaluator

    def convert(self, format_type, metaparam, name=None, vb=True, via=None):
        """
        Converts to a different parametrization

        Parameters
        ----------
        format_type: string
            format of the target parametrization
        metaparam: integer or numpy.ndarray, float
            metaparameter(s) for the target parametrization
        name: string, optional
            name of the target parametrization
        vb: boolean, optional
            report progress to stdout?
        via: dict, optional
            keywords for intermediate format

        Returns
        -------
        parametrization: qp.Parametrization object
            the target parametrization object

        Notes
        -----
        TO DO: Add in keywords in via for complicated conversion options
        """
        if name is None:
            name = self.name

        if format_type == 'funcform':
            intermediate = qp.Parametrization.FuncForm(metaparam, vb=vb)
            target = intermediate.convert('funcform', metaparam, name=name, vb=vb)
        elif format_type == 'pointeval':
            try:
                evaluator = self.pdf_evaluator
            except AttributeError:
                evaluator = self._pdf()
            param = evaluator(metaparam)
            target = PointEval(metaparam, param, name=name, vb=vb)
        elif format_type == 'quantile':
            try:
                evaluator = self.ppf_evaluator
            except AttributeError:
                evaluator = self._ppf()
            param = evaluator(metaparam)
            target = Quantile(metaparam, param, name=name, vb=vb)
        elif format_type == 'sample':
            try:
                evaluator = self.rvs_evaluator
            except AttributeError:
                evaluator = self._rvs()
            param = evaluator(metaparam)
            target = Sample(param, name=name, vb=vb)
        else:
            print 'Conversion to ' + format_type + ' not supported'
            return

        return target# new_parametrization

class Quantile(Parametrization):
    """
    A parametrization of the quantile format
    """
    def __init__(self, metaparam, param, name=None, vb=True):
        Parametrization.__init__(self, 'quantile', metaparam, param, name=name, vb=vb)

class Sample(Parametrization):
    """
    A parametrization of the sample format
    """
    def __init__(self, param, metaparam=None, name=None, vb=True):
        Parametrization.__init__(self, 'sample', len(param), param, name=name, vb=vb)

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
