from typing import Sequence, Callable, Mapping, Union, List, Any, Tuple, Dict

import pandas as pd
from more_itertools import collapse
import numpy as np


def flatten(data, remaining_levels: int = 0) -> List[Union[Any, str]]:
    """
    transform this:
    {'lc1': [
       ['inductor1.i', 'inductor2.i', 'inductor3.i'],
       ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']],
     'lcl1': [
        ['inductor1.i', 'inductor2.i', 'inductor3.i'],
        ['capacitor1.v', 'capacitor2.v', 'capacitor3.v']]}

    to:
    ['lc1.inductor1.i', 'lc1.inductor2.i', 'lc1.inductor3.i',
     'lc1.capacitor1.v', 'lc1.capacitor2.v', 'lc1.capacitor3.v',
     'lcl1.inductor1.i', 'lcl1.inductor2.i', 'lcl1.inductor3.i',
     'lcl1.capacitor1.v', 'lcl1.capacitor2.v', 'lcl1.capacitor3.v']
    or:
    [['lc1.inductor1.i', 'lc1.inductor2.i', 'lc1.inductor3.i'],
     ['lc1.capacitor1.v', 'lc1.capacitor2.v', 'lc1.capacitor3.v'],
     ['lcl1.inductor1.i', 'lcl1.inductor2.i', 'lcl1.inductor3.i'],
    ['lcl1.capacitor1.v', 'lcl1.capacitor2.v', 'lcl1.capacitor3.v']]


    :param data:
    :param remaining_levels:
    :return:
    """
    # collapse outer dicts
    if isinstance(data, dict):
        # flatten all the dicts
        df = pd.json_normalize(data)
        data = df.to_dict(orient='records')[0]
        # move the key into the lists
        for k, v in data.items():
            data[k] = nested_map(v, lambda suffix: '.'.join([k, suffix]))
        data = list(data.values())
    # count levels and collapse to keep the levels as needed
    depth = nested_depth(data)
    if remaining_levels is None:
        remaining_levels = depth - 1
    return list(collapse(data, levels=depth - remaining_levels - 1))


def nested_map(l: Union[list, tuple, Mapping, np.ndarray], fun: Callable):
    """


    :param l:
    :param fun:
    :return:
    """
    if isinstance(l, Mapping):
        return {k: nested_map(v, fun) for k, v in l.items()}
    if isinstance(l, (list, tuple)):
        return [nested_map(l_, fun) for l_ in l]
    if isinstance(l, np.ndarray):
        # empty_like would keep the datatype, with empty,
        # we enforce that the dtype is infered from the result of the mapping function
        a = np.empty(l.shape)
        for idx in np.ndindex(l.shape):
            a[idx] = nested_map(l[idx], fun)
        return a
    return fun(l)


def nested_depth(l: Union[Any, List, Tuple]) -> int:
    """
    Calculate the maximum depth of a nested sequence.

    :param l: nested sequence. The containing data structures are currently restricted to lists and tuples,
     because allowing any sequence would also result in traversing strings for example
    :return: maximum depth
    """
    if isinstance(l, (list, tuple, set)):
        if l:
            # if the list contains elements
            return 1 + max((nested_depth(l_) for l_ in l))
        return 1
    return 0


def fill_params(template, data: Union[pd.Series, Mapping]):
    """
    Uses a template, that can be traversed by nested_map.
    Each entry in the template, that is a key in the mapping is replaced by the value it is mapped to.

    :param template: template containing keys
    :param data: mapping of keys to values
    :return:
    """
    if isinstance(data, pd.Series):
        data = data.to_dict()
    elif not isinstance(data, Mapping):
        raise ValueError("must be a mapping")

    # keep key if there is no substitute
    return nested_map(template, lambda k: data.get(k, k))
