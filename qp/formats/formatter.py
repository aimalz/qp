import metaformatter.MetaFormatter as MetaFormatter

class Formatter(object):
    __metaclass__ = MetaFormatter

    def __init__(self, metaparam, param, res=None):
        print ("calling Formatter.__init__()")
        self.metaparam = metaparam
        self.param = param
        self.res = res

    @classmethod
    def _set_easy(cls):
        cls.easy_to = []
        cls.easy_from = []
        print(cls.easy_to, cls.easy_from)
        return cls.graph

    @classmethod
    def _build_graph(cls):
        print("calling _build_graph" + str(cls) + " name: " + cls.name)
        if cls.easy_to != []:
            cls.graph[cls.name.lower()] = set().union(cls.graph[cls.name], cls.easy_to)
        for i in cls.easy_from:
            cls.graph[i] = set().union(cls.graph[i], [cls.name])
        return

    def call(self, key, new_metaparam, **kwargs):
        if key == 'cdf':
            return self._cdf(new_metaparam, kwargs)
        if key == 'fit':
            return self._fit(new_metaparam, kwargs)
        if key == 'pdf':
            return self._pdf(new_metaparam, kwargs)
        if key == 'ppf':
            return self._ppf(new_metaparam, kwargs)
        if key == 'rvs':
            return self._rvs(new_metaparam, kwargs)

    def _cdf(self, metaparam, **kwargs):
        """
        metaparam is array of points at which to evaluate
        """
        print('CDF not yet implemented for '+self.name)
        return None

    def _fit(self, metaparam, **kwargs):
        """
        metaparam is function to fit to
        """
        print('FIT not yet implemented for '+self.name)
        return None

    def _pdf(self, metaparam, **kwargs):
        """
        metaparam is array of points at which to evaluate
        """
        print('PDF not yet implemented for '+self.name)
        return None

    def _ppf(self, metaparam, **kwargs):
        """
        metaparam is array of probabilities at which to evaluate
        """
        print('PPF not yet implemented for '+self.name)
        return None

    def _rvs(self, metaparam, **kwargs):
        """
        metaparam is integer of samples
        """
        print('RVS not yet implemented for '+self.name)
        return None
