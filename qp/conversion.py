"""This module implements tools to convert between distributions"""

import sys

from .ensemble import Ensemble

from .dict_utils import get_val_or_default, set_val_or_default, pretty_print


PRINT_PREFIXS = ["To ", "  From ", "    Method "]


class ConversionDict:
    """Dictionary of possible conversions

    Notes
    -----
    This dictionary is implemented as a triply nested dictionary,

    dict[class_to][class_from][method]

    At each level, `None` is used define the default behavior,
    i.e.,  the behavior if the additional arguements are not found
    in the dictionary.
    """

    def __init__(self):
        """Class constructor, builds an empty dictionary"""
        self._conv_dict = {}

    def _get_convertor(self, class_to, class_from, method=None):

        to_dict = get_val_or_default(self._conv_dict, class_to)
        if to_dict is None:
            return None
        to_from_dict = get_val_or_default(to_dict, class_from)
        if to_from_dict is None:
            return None
        func = get_val_or_default(to_from_dict, method)
        return func


    def _convert_ensemble(self, ensemble_from, class_to, method=None, **kwargs):

        class_from = ensemble_from.gen_class
        convert = self._get_convertor(class_to, class_from, method)
        return convert(ensemble_from, class_to, **kwargs)


    def convert(self, obj_from, class_to, method=None, **kwargs):
        """Convert a distribution or ensemble

        Parameters
        ----------
        obj_from :  `scipy.stats.rv_continuous or qp.ensemble`
            Input object
        class_to : sub-class of `scipy.stats.rv_continuous`
            The class we are converting to
        method : `str`
            Optional argument to specify a non-default conversion algorithm
        kwargs : keyword arguments are passed to the output class constructor

        Notes
        -----
        If obj_from is a single distribution this will return a single distribution of
        type class_to.

        If obj_from is a `qp.Ensemble` this will return a `qp.Ensemble` of distributions
        of type class_to.
        """
        if isinstance(obj_from, Ensemble):
            return self._convert_ensemble(obj_from, class_to, method, **kwargs)
        raise TypeError("Tried to convert object of type %s" % type(obj_from))


    def add_mapping(self, func, class_to, class_from, method=None):
        """Add a mapping to this dictionary

        Parameters
        ----------
        func : `function`
            The function used to do the conversion
        class_to : sub-class of `scipy.stats.rv_continuous`
            The class we are converting to
        class_from :  sub-class of `scipy.stats.rv_continuous`
            The class we are converting from
        method : `str`
            Optional argument to specify a non-default conversion algorithm
        """
        to_dict = set_val_or_default(self._conv_dict, class_to, {})
        to_from_dict = set_val_or_default(to_dict, class_from, {})
        ret_func = set_val_or_default(to_from_dict, method, func)
        return ret_func

    def pretty_print(self, stream=sys.stdout):
        """Print a level of the converstion dictionary in a human-readable format

        Parameters
        ----------
        stream : `stream`
            The stream to print to
        """
        pretty_print(self._conv_dict, PRINT_PREFIXS, stream=stream)


CONVERSIONS = ConversionDict()

def qp_convert(obj_from, class_to, method=None, **kwargs):
    """Convert a distribution or ensemble

    Parameters
    ----------
    obj_from :  `scipy.stats.rv_continuous or qp.ensemble`
        Input object
    class_to : sub-class of `scipy.stats.rv_continuous`
        The class we are converting to
    method : `str`
        Optional argument to specify a non-default conversion algorithm
    kwargs : keyword arguments are passed to the output class constructor

    Notes
    -----
    If obj_from is a single distribution this will return a single distribution of
    type class_to.

    If obj_from is a `qp.Ensemble` this will return a `qp.Ensemble` of distributions
    of type class_to.
    """
    return CONVERSIONS.convert(obj_from, class_to, method, **kwargs)


def qp_add_mapping(func, class_to, class_from, method=None):
    """
    Parameters
    ----------
    func : `function`
        The function used to do the conversion
    class_to : sub-class of `scipy.stats.rv_continuous`
        The class we are converting to
    class_from :  sub-class of `scipy.stats.rv_continuous`
        The class we are converting from
    method : `str`
        Optional argument to specify a non-default conversion algorithm
    """
    return CONVERSIONS.add_mapping(func, class_to, class_from, method)
