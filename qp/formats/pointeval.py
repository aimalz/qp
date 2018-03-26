from parametrization import Parametrization
from funcform import FuncForm

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

        self.last_fit_args = None
        self.last_pdf_scheme = None
        self.last_pdf_args = None
        self.last_cdf_scheme = None
        self.last_cdf_args = None

    def _cdf(self, scheme, **cdf_args):
        """
        Returns a function that gives the integral from the minimum evaluation point to specified value
        """
        self.last_cdf_scheme = scheme
        self.last_cdf_args = cdf_args
        if scheme == 'spline':
            xi = min(self.metaparam)
            def cdf(xf):
                fun = spi.InterpolatedUnivariateSpline(self.metaparam, self.param, fit_args).integral
                return fun(xi, xf)
        else:
            print scheme+' not yet supported'
            return
        self.cdf_evaluator = np.vectorize(cdf)
        return self.cdf_evaluator

    def _fit(self, **fit_args):
        """
        Returns a function that fits a functional form via optimization

        Notes
        -----
        TO DO: add a way to actually feed in fit arguments
        """
        self.last_fit_args = fit_args
        def fit(metaparam, ivals=None):
            if ivals is None:
                ivals = {}
                intermediate = metaparam['function']()
                shape, loc, scale = intermediate.dist._parse_args(*intermediate.args, **intermediate.kwds)
                if len(shape) != 0:
                    ivals['shape'] = shape
                ivals['loc'] = loc
                ivals['scale'] = scale
            def fun(x, **param):
                ff = FuncForm(metaparam, param)
                pe = ff.convert('pointeval', x)
                y = pe.param
                return y
            fmodel = lmfit.Model(fun)
            pars = fmodel.make_params(**ivals)
            res = fmodel.fit(self.param, params=pars, x=self.metaparam)
            return res.values
        self.fit_evaluator = fit
        return self.fit_evaluator

    def _pdf(self, scheme, **pdf_args):
        """
        Returns a non-parametric PDF function
        """
        self.last_pdf_scheme = scheme
        self.last_pdf_args = pdf_args
        if scheme == 'interp1d':
            def pdf(x):
                fun = spi.interp1d(self.metaparam, self.param, pdf_args)
                return fun(x)
        if scheme == 'spline':
            def pdf(x):
                fun = spi.InterpolatedUnivariateSpline(self.metaparam, self.param, pdf_args)
                return fun(x)
        self.pdf_evaluator = pdf
        return self.pdf_evaluator

    def _ppf(self, scheme, **ppf_args):
        """
        Returns a function that gives the PPF
        """
        if self.last_cdf_scheme == scheme and self.last_cdf_args == ppf_args:
            try:
                cdf = self.cdf_evaluator
            except AttributeError:
                cdf = self._cdf(scheme, ppf_args)
        else:
            cdf = self._cdf(scheme, ppf_args)
        if scheme == 'spline':
            def ppf(q):
                iy = cdf(self.metaparam)
                fun = spi.InterpolatedUnivariateSpline(iy, self.metaparam, ppf_args)
                return fun(q)
        else:
            print scheme+' not yet supported'
            return
        self.ppf_evaluator = ppf
        return self.ppf_evaluator

    def _rvs(self, scheme, **rvs_args):
        """
        Returns a function giving samples
        """
        if self.last_pdf_scheme == scheme and self.last_pdf_args == rvs_args:
            try:
                pdf = self.pdf_evaluator
            except AttributeError:
                pdf = self._pdf(scheme, rvs_args)
        else:
            pdf = self._pdf(scheme, rvs_args)
        (xmin, xmax) = (min(self.metaparam), max(self.metaparam))
        (ymin, ymax) = (min(self.param), max(self.param))
        (xran, yran) = (xmax - xmin, ymax - ymin)
        def rvs(N):
            xs = []
            while len(samples) < N:
                (x, y) = (xmin + xran * np.random.uniform(), ymin + yran * np.random.uniform())
                if y < pdf(x):
                    xs.append(x)
            return xs
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
        TO DO: fix these try/excepts in if/elses
        TO DO: enable _fit arguments
        """
        if vb: print 'PointEval converting to '+format_type+' with metaparam '+str(metaparam)
        if name is None:
            name = self.name

        if format_type == 'funcform':
            try:
                evaluator = self.fit_evaluator
            except AttributeError:
                evaluator = self._fit()
            param = evaluator(metaparam)
            target = FuncForm(metaparam, param, name=name, vb=vb)
        elif format_type == 'pointeval':
            if via is None:
                via = {'scheme': 'interp1d', 'fit_args': None}
            if 'format_type' in via:
                if vb:
                    print 'Converting to '+format_type+' via '+via['format_type']
                intermediate = self.convert(via['format_type'], via['metaparam'], vb=vb, **via['fit_args'])
                target = intermediate.convert(format_type, metaparam, name=Name, vb=vb)
            elif 'scheme' in via:
                if vb:
                    print 'Converting to '+format_type+' via '+via['scheme']
                if self.last_fit_scheme == via['scheme'] and self.last_fit_args == via['fit_args']:
                    try:
                        evaluator = self.pdf_evaluator
                    except AttributeError:
                        evaluator = self._pdf(via['scheme'], **via['fit_args'])
                else:
                    evaluator = self._pdf(via['scheme'], **via['fit_args'])
                param = evaluator(metaparam)
                target = PointEval(metaparam, param, name=name, vb=vb)
        elif format_type == 'quantile':
            if via is None:
                via = {'scheme': 'spline', 'fit_args': None}
            if 'format_type' in via:
                if vb:
                    print 'Converting to '+format_type+' via '+via['format_type']
                intermediate = self.convert(via['format_type'], via['metaparam'], vb=vb)
                target = intermediate.convert(format_type, metaparam, name=Name, vb=vb)
            elif 'scheme' in via:
                if vb:
                    print 'Converting to '+format_type+' via '+via['scheme']
                if via['scheme'] == 'spline':
                    if self.last_fit_scheme == via['scheme'] and self.last_fit_args == via['fit_args']:
                        try:
                            evaluator = self.ppf_evaluator
                        except AttributeError:
                            evaluator = self._ppf(via['scheme'], via['fit_args'])
                    else:
                        evaluator = self._ppf(via['scheme'], via['fit_args'])
                param = evaluator(metaparam)
                target = Quantile(metaparam, param, name=name, vb=vb)
        elif format_type == 'sample':
            try:
                evaluator = self.rvs_evaluator
            except AttributeError:
                evaluator = self._rvs(via['scheme'], via['fit_args'])
            param = evaluator(metaparam)
            target = Sample(param, name=name, vb=vb)
        else:
            print 'Conversion to ' + format_type + ' not supported'
            return

        return target# new_parametrization
