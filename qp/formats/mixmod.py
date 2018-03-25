import numpy as np
import scipy as sp
from scipy import stats as sps
import scipy.optimize as op

import formatter.Formatter as Formatter

class MixMod(Formatter):
    def __init__(self, metaparam, param):
        # metaparam is list of scipy.stats.rv_continuous objects
        # param is list of dicts with coefficient and params (which is a dict itself)
        super(MixMod, self).__init__(metaparam, param)

        self.N_component = len(self.metaparam)
        self.component_range = range(self.N_component)

        self.res = sum([1 + len(self.param[i]['param'].keys()) for i in self.component_range])

        coefficients = [self.param[i]['coefficient'] for i in self.component_range]
        self.coefficients = coefficients / np.sum(coefficients)
        each_param = [self.param[i]['param'] for i in self.component_range]

        self.functions = [self.metaparam[i](**each_param[i]) for i in self.component_range]
#         coefficients = np.array([mp['coefficient'] for mp in self.metaparam])
#         self.functions = np.array([mp['function'] for mp in self.metaparam])

    @classmethod
    def _set_easy(cls):
#         print ('calling MixMod._set_easy()')
        cls.easy_to = ['cdf', 'pdf', 'rvs']
        cls.easy_from = ['fit']

    def _cdf(self, metaparam):
        """
        metaparam is array of points at which to evaluate
        """
        if metaparam is None:
            randos = self._rvs(None)
            metaparam = np.linspace(min(randos), max(randos), self.res)
        param = np.zeros(np.shape(metaparam))
        for c in self.component_range:
            param += self.coefficients[c] * self.functions[c].cdf(metaparam)
        return (metaparam, param)

    def _pdf(self, metaparam):
        """
        metaparam is array of points at which to evaluate
        """
        if metaparam is None:
            randos = self._rvs(None)
            metaparam = np.linspace(min(randos), max(randos), self.res)
        param = np.zeros(np.shape(metaparam))
        for c in self.component_range:
            param += self.coefficients[c] * self.functions[c].pdf(metaparam)
        return (metaparam, param)

    def _ppf(self, metaparam, ivals=None, **kwargs):
        """
        metaparam is array of probabilities at which to evaluate
        """
        print('PPF not recommended; performing optimization with '+str(kwargs))
        if metaparam is None:
            metaparam = np.linspace(1. / (self.res + 1.), 1., self.res, endpoint=False)
        if 'method' not in kwargs:
            kwargs['method'] = 'Nelder-Mead'
        if 'options' not in kwargs:
            kwargs['options'] = {"maxfev": 1e5, "maxiter":1e5}
        if 'tol' not in kwargs:
            kwargs['tol'] = 1e-8
        if ivals is not None:
            param0 = ivals
        else:
            all_cdfs = np.zeros(self.res)
            for c in self.component_range:
                all_cdfs += self.functions[c].ppf(metaparam)
            param0 = all_cdfs / self.n_components
        param = np.zeros(self.res)
        for n in range(self.res):
            def ppf_helper(x):
                return np.absolute(metaparam[n] - self._cdf(x))
            res = spo.minimize(ppf_helper, param0[n], kwargs)
            param[n] += res.x
        return (metaparam, param)

    def _rvs(self, metaparam):
        """
        metaparam is integer of samples
        """
        if metaparam is None:
            metaparam = self.res
        groups = np.random.choice(self.component_range, metaparam, p=self.coefficients)
        u, counts = np.unique(groups, return_counts=True)
        samples = np.empty(0)
        for i in range(len(u)):
            samples = np.append(samples, self.functions[u[i]].rvs(counts[i]))
        param = np.array(samples).flatten()
        return (metaparam, param)
