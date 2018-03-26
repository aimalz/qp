class Sample(Parametrization):
    """
    A parametrization of the sample format
    """
    def __init__(self, param, metaparam=None, name=None, vb=True):
        Parametrization.__init__(self, 'sample', len(param), param, name=name, vb=vb)
