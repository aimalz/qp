class Quantile(Parametrization):
    """
    A parametrization of the quantile format
    """
    def __init__(self, metaparam, param, name=None, vb=True):
        Parametrization.__init__(self, 'quantile', metaparam, param, name=name, vb=vb)

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
        """
