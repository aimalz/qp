"""This module implements tools to persist and read back ensembles"""

import sys
from .dict_utils import get_val_or_default, set_val_or_default, pretty_print


READER_PREFIXS = ["Class ", "  Version "]
WRITER_PREFIXS = ["Class ", "  Version "]


class ReaderDict:
    """Dictionary of ensemble reader functions

    Notes
    -----
    This dictionary is implemented as a doubly nested dictionary,

    dict[class_to][version]

    At each level, `None` is used define the default behavior,
    i.e.,  the behavior if the additional arguements are not found
    in the dictionary.
    """

    def __init__(self):
        """Class constructor, builds an empty dictionary"""
        self._conv_dict = {}

    def get_reader(self, class_to, version=None):
        """Get the reader that can be used to construct an ensemble

        Parameters
        ----------
        class_to : `str`
            The name of the class we will read the data into
        version : `str`
            Optional argument to specify a non-default version of the reder
        kwargs : keyword arguments are passed to the output class constructor

        Returns
        -------
        writer : `function`
            The function that will write data of this class
        """
        to_dict = get_val_or_default(self._conv_dict, class_to)
        if to_dict is None:
            return None
        func = get_val_or_default(to_dict, version)
        return func


    def add_mapping(self, func, class_to, version=None):
        """Add a mapping to this dictionary

        Parameters
        ----------
        func : `function`
            The function used to do the conversion
        class_to :  `str`
            The class we are converting the data into
        version : `str`
            Optional argument to specify a non-default conversion algorithm
        """
        to_dict = set_val_or_default(self._conv_dict, class_to, {})
        ret_func = set_val_or_default(to_dict, version, func)
        return ret_func

    def pretty_print(self, stream=sys.stdout):
        """Print a level of the converstion dictionary in a human-readable format

        Parameters
        ----------
        stream : `stream`
            The stream to print to
        """
        pretty_print(self._conv_dict, READER_PREFIXS, idx=0, stream=stream)




class WriterDict:
    """Dictionary of ensemble writer functions

    Notes
    -----
    This dictionary is implemented as a doubly nested dictionary,

    dict[class_from][version]

    At each level, `None` is used define the default behavior,
    i.e.,  the behavior if the additional arguements are not found
    in the dictionary.
    """

    def __init__(self):
        """Class constructor, builds an empty dictionary"""
        self._conv_dict = {}

    def get_writer(self, class_from, version=None):
        """Get the writer that can be used to convert an ensemble

        Parameters
        ----------
        class_from : sub-class of `scipy.stats.rv_continuous`
            The class we are writing
        version : `str`
            Optional argument to specify a non-default version of the writer
        kwargs : keyword arguments are passed to the output class constructor

        Returns
        -------
        writer : `function`
            The function that will write data of this class
        """
        from_dict = get_val_or_default(self._conv_dict, class_from)
        if from_dict is None:
            return None
        func = get_val_or_default(from_dict, version)
        return func

    def add_mapping(self, func, class_from, version=None):
        """Add a mapping to this dictionary

        Parameters
        ----------
        func : `function`
            The function used to do the conversion
        class_from :  sub-class of `scipy.stats.rv_continuous`
            The class we are converting from
        version : `str`
            Optional argument to specify a non-default conversion algorithm
        """
        from_dict = set_val_or_default(self._conv_dict, class_from, {})
        ret_func = set_val_or_default(from_dict, version, func)
        return ret_func

    def pretty_print(self, stream=sys.stdout):
        """Print a level of the converstion dictionary in a human-readable format

        Parameters
        ----------
        stream : `stream`
            The stream to print to
        """
        pretty_print(self._conv_dict, WRITER_PREFIXS, idx=0, stream=stream)



READERS = ReaderDict()
WRITERS = WriterDict()



def add_reader_mapping(func, class_to, version=None):
    """
    Add a mapping to the `qp.ReaderDict` dictionary of
    classes to read persistent data.

    Parameters
    ----------
    func : `function`
        The function used to do the conversion
    class_to : `str`
        The name of the class we are converting to
    version : `int`
        Optional argument to specify the version of the persistent data
    """
    return READERS.add_mapping(func, class_to, version)

def register_pdf_class(cls):
    """
    Register a class with the `qp.ReaderDict` dictionary of
    classes to read persistent data.

    Parameters
    ----------
    cls : `class`
        The class we are registring
    """

    for i in range(cls.version+1):
        add_reader_mapping(cls, cls.name, i)


def add_writer_mapping(func, class_from, version=None):
    """
    Add a mapping to the `qp.WriterDict` dictionary of
    classes to read persistent data.

    Parameters
    ----------
    func : `function`
        The function used to do the conversion
    class_from :  sub-class of `scipy.stats.rv_continuous`
        The class we are writing from
    version : `int`
        Optional argument to specify the version of the persistent data
    """
    return WRITERS.add_mapping(func, class_from, version)


def get_qp_reader(class_to, version=None):
    """
    Get the reader used to read presisent data for a particular type of PDF

    Parameters
    ----------
    class_to : `str`
        The name of the class we are converting to
    version : `int`
        Optional argument to specify the version of the persistent data

    Returns
    -------
    val : 'function` or `qp.Pdf` subclass
        The object that will covert the persistent data to `qp.Ensemble`
    """
    val = READERS.get_reader(class_to, version)
    if val is None:
        print(class_to, version)
        READERS.pretty_print()
    return val

def get_qp_writer(cls, version=None):
    """
    Get the writer used to write presisent data for a particular type of PDF

    Parameters
    ----------
    cls : `class`
        The name of the class we are persisting
    version : `int`
        Optional argument to specify the version of the persistent data

    Returns
    -------
    val : 'function` or `qp.Pdf` subclass
        The object that will persist the `qp.Ensemble`
    """
    return WRITERS.get_writer(cls, version)
