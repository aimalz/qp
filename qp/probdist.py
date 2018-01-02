class ProbDist(object):
    """
    An object that is defined by its parametrizations of a single PDF
    """
    def __init__(self, parametrizations=None, base=None, vb=True):
        """
        Parameters
        ----------
        parametrizations: list, qp.Parametrization objects, optional
            list of parametrizations to initialize with
        base: string, optional
            name of base (true) parametrization
        vb: boolean, optional
            print progress to stdout?
        """
        self.base = base
        self.parametrizations = {}
        if parametrizations is not None:
            if self.base is None:
                self.base = parametrizations[0].name
            for i in range(len(parametrizations)):
                self.parametrizations[parametrizations[i].name] = parametrizations[i]
        elif vb:
            print 'Warning: initializing a ProbDist object without any parametrizations!'

    def add(self, parametrization, base=False):
        """
        Adds another parametrization to the object

        Parameters
        ----------
        parametrization: qp.Parametrization object
            parametrization to add
        base: boolean, optional
            set this parametrization as the base (true) one?
        """
        self.parametrizations[parametrization.name] = parametrization
        if base or self.base is None:
            self.base = prametrization.name
        return

    def convert(self, target_format, target_metaparameters, target_name=None, origin=None, vb=True, **kwargs):
        """
        Converts from a given parametrization to a new one

        Parameters
        ----------
        target_format: string
            format of the target parametrization
        target_metaparameters: integer or numpy.ndarray, float
            metaparameters of the target parametrization
        target_name: string, optional
            name to give to the target parametrization
        origin: qp.Parametrization object, optional
            parametrization to convert from defaults to base if set
        vb: boolean, optional
            print progress to stdout?
        **kwargs: dict, optional
            keywords for the conversion

        Returns
        -------
        target: qp.Parametrization object
            target parametrization

        Notes
        -----
        If there is no name provided, it may overwrite an existing parametrization with the same format.
        """
        if origin is None:
            origin = self.base
        target = self.parametrizations[origin].convert(target_format, target_metaparameters, target_name, **kwargs)
        self.add(target)
        return target

    def compare(self, metric, approx_parametrization, base_parametrization=None, **kwargs):
        """
        Computes a metric between two representations

        Parameters
        ----------
        metric: function
            function of two parametrizations
        approx_parametrization: string
            name of approximate parametrization
        base_parametrization: string, optional
            name of base parametrization
        **kwargs: dict, optional
            additional keyword arguments needed by metric

        Returns
        -------
        val: float
            the value of the metric between the input parametrizations
        """
        if base_parametrization is None:
            base_parametrization == self.base
        base = self.parametrizations[base_parametrization]
        approx = self.parametrizations[approx_parametrization]
        return metric(base, approx, **kwargs)
