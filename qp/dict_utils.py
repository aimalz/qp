"""This module implements tools to convert between distributions"""

import sys


def get_val_or_default(in_dict, key):
    """Helper functions to return either an item in a dictionary or the default value of the dictionary

    Parameters
    ----------
    in_dict : `dict`
        input dictionary
    key : `str`
        key to search for

    Returns
    -------
    out : `dict` or `function`
        The requested item

    Notes
    -----
    This will first try to return:
        in_dict[key] : i.e., the requested item.
    If that fails it will try
        in_dict[None] : i.e., the default for that dictionary.
    If that failes it will return
        None
    """
    if key in in_dict:
        return in_dict[key]
    if None in in_dict:
        return in_dict[None]
    return None


def set_val_or_default(in_dict, key, val):
    """Helper functions to either get and item from or add an item to a dictionary and return that item

    Parameters
    ----------
    in_dict : `dict`
        input dictionary
    key : `str`
        key to search for
    val : `dict` or `function`
        item to add to the dictionary

    Returns
    -------
    out : `dict` or `function`
        The requested item

    Notes
    -----
    This will first try to return:
        in_dict[key] : i.e., the requested item.
    If that fails it will try
        in_dict[None] : i.e., the default for that dictionary.
    If that failes it will return
        None
    """
    if key in in_dict:
        return in_dict[key]
    in_dict[key] = val
    return val


def pretty_print(in_dict, prefixes, idx=0, stream=sys.stdout):
    """Print a level of the converstion dictionary in a human-readable format

    Parameters
    ----------
    in_dict : `dict`
        input dictionary
    prefixs : `list`
        The prefixs to use at each level of the printing
    idx : `int`
        The level of the input dictionary we are currently printing
    stream : `stream`
        The stream to print to
    """
    prefix = prefixes[idx]
    for key, val in in_dict.items():
        if key is None:
            key_str = "default"
        else:
            key_str = key
        if isinstance(val, dict):
            stream.write("%s%s:\n" % (prefix, key_str))
            pretty_print(val, prefixes, idx+1, stream)
        else:
            stream.write("%s%s : %s\n" % (prefix, key_str, val))


def print_dict_shape(in_dict):
    """Print the shape of arrays in a dictionary.
    This is useful for debugging `astropy.Table` creation.

    Parameters
    ----------
    in_dict : `dict`
        The dictionary to print
    """
    for key, val in in_dict.items():
        print(key, val.shape)


def slice_dict(in_dict, subslice):
    """Create a new `dict` by taking a slice of of every array in a `dict`

    Parameters
    ----------
    in_dict : `dict`
        The dictionary to conver
    subslice : `int` or `slice`
        Used to slice the arrays

    Returns
    -------
    out_dict : `dict`
        The converted dicionary
    """

    out_dict = {}
    for key, val in in_dict.items():
        try:
            out_dict[key] = val[subslice]
        except (KeyError, TypeError):
            out_dict[key] = val
    return out_dict
