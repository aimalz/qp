import numpy as np
import scipy.interpolate as spi
import scipy.stats as sps

class Parametrization(object):
    """
    An object defining a general parametrization of a PDF
    """
    def __init__(self, base_format, metaparam, param, name=None, vb=True):
        """
        Parameters
        ----------
        base_format: qp.Formatter object
            the format function
        metaparam: numpy.ndarray, float
            the parameters of the format (that could be shared across a qp.Ensemble object)
        param: numpy.ndarray, float
            the parameters unique to the qp.ProbDist object
        name: string, optional
            a name for the parametrization if multiple parametrizations of the same format are to be associated with a single PDF object
        vb: boolean, optional
            report on progress to stdout?

        Notes
        -----
        Includes default behavior for a generic format, currently error messages to be overridden by subclasses
        """
        self.base_format = base_format
        self.metaparam = metaparam
        self.param = param
        if name is None:
            self.name = self.format
        else:
            self.name = name

    def convert(self, target_format, metaparam, name=None, vb=True, via=None, **kwargs):
        """
        Converts to a different parametrization

        Parameters
        ----------
        target: qp.Formatter object
            format of the target parametrization
        metaparam: integer or numpy.ndarray, float or dict
            metaparameter(s) for the target parametrization
        name: string, optional
            name of the target parametrization
        vb: boolean, optional
            report on progress to stdout?
        via: dict, optional
            keyword arguments for conversion through an intermediate parametrization
        kwargs: dict, optional
            optional keyword arguments for the conversion method

        Returns
        -------
        parametrization: qp.Parametrization object
            the target parametrization object
        """
        assert target_format.name.lower() in target_format.registry
        if target_format == 'funcform':
            if self.fit_evaluator is None:
                evaluator = self._fit(kwargs)
            else:
                evaluator = self.fit_evaluator
            param = evaluator(metaparam)
            target = FuncForm(metaparam, param, name=name, vb=vb)
        if target_format == 'pointeval':
            if self.pdf_evaluator is None:
                evaluator = self._pdf(kwargs)
            else:
                evaluator = self.pdf_evaluator
            param = evaluator(metaparam)
            target = PointEval(metaparam, param, name=name, vb=vb)
        if target_format == 'quantile':
            if self.ppf_evaluator is None:
                evaluator = self._ppf(kwargs)
            else:
                evaluator = self.ppf_evaluator
            param = evaluator(metaparam)
            target = Quantile(metaparam, param, name=name, vb=vb)
        if target_format == 'sample':
            if self.rvs_evaluator is None:
                evaluator = self._rvs(kwargs)
            else:
                evaluator = self.rvs_evaluator
            param = evaluator(metaparam)
            target = Sample(metaparam, param, name=name, vb=vb)
        return target# new_parametrization
