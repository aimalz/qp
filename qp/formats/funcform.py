from formats import Parametrization

class FuncForm(Parametrization):
    """
    A parametrization of a format with a functional form
    """
    def __init__(self, metaparam, param, name=None, vb=True):
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
        Parametrization.__init__(self, 'funcform', metaparam, param, name=name, vb=vb)

    def _pdf(self, **kwargs):
        """
        Makes a function that evaluates the PDF of a functional form
        """
        def pdf(x):# x points
            y = self.metaparameters['rv_continuous'](loc=self.param['loc'], scale=self.parameters['scale'])
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
        params = self.metaparam.copy()
        params.update(self.param)
        params.pop('function')
        self.params = params
        use_params = params.copy()
        if 'shape' in params.keys() and params['shape'] is not None and len(params['shape']) == 0:
            defaults = params['shape']
            use_params.pop('shape')
            self.representation = self.function(*defaults, **use_params)
        else:
            self.representation = self.function(**use_params)

    def _cdf(self):
        """
        Makes a function that evaluates the CDF of a functional form
        """
        def cdf(x):# x points
            return self.representation.cdf(x)
        self.cdf_evaluator = cdf
        return self.cdf_evaluator

    def _fit(self):
        """
        Returns a function that fits to a different functional form
        """
        def fit(x):# x points
            return self.representation.fit(x)
        self.fit_evaluator = fit
        return self.fit_evaluator

    def _pdf(self):
        """
        Makes a function that evaluates the PDF of a functional form
        """
        def pdf(x):# x points
            return self.representation.pdf(x)
        self.pdf_evaluator = pdf
        return self.pdf_evaluator

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
            target = intermediate.convert(format_type, metaparam, name=name, vb=vb)
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
