from typing import Callable, Mapping, Union, Any, List

import numpy as np
import pandas as pd
from more_itertools import collapse


def flatten(data: Union[dict, list], remaining_levels: int = 0) -> list:
    """
    transform this:

    >>> {'a': {'b': [['i', 'v'],
    >>>              ['k', 'h']]}}

    results into

    >>> ['a.b.i', 'a.b.v', 'a.b.k', 'a.b.h']

    or

    >>> [['a.b.i', 'a.b.v'], ['a.b.k', 'a.b.h']]


    :param data: data to flatten. A nested dictionary containing nested lists as keys in the lowest level.
    :param remaining_levels: number of levels to preserve in the nested list
    :return: Flattened data as a nesteted list
    """
    # collapse outer dicts
    if isinstance(data, dict):
        # flatten all the dicts
        df = pd.json_normalize(data)
        data = df.to_dict(orient='records')[0]
        # move the key into the lists
        for k, v in data.items():
            data[k] = nested_map(lambda suffix: '.'.join([k, suffix]), v)
        data = list(data.values())
    # count levels and collapse to keep the levels as needed
    depth = nested_depth(data)
    if remaining_levels is None:
        remaining_levels = depth - 1
    return list(collapse(data, levels=depth - remaining_levels - 1))


def nested_map(fun: Callable, structure: Union[list, tuple, Mapping, np.ndarray]) \
        -> Union[list, tuple, Mapping, np.ndarray]:
    """
    Traverses data structure and substitutes every element with the result of the callable

    :param fun: Callable to be applied to every value
    :param structure: Nesting of dictionaries or lists. For mappings, the callable is applied to the values.
    :return:
    """
    if isinstance(structure, Mapping):
        return {k: nested_map(fun, v) for k, v in structure.items()}
    if isinstance(structure, (list, tuple)):
        return [nested_map(fun, l_) for l_ in structure]
    if isinstance(structure, np.ndarray):
        # empty_like would keep the datatype, with empty,
        # we enforce that the dtype is infered from the result of the mapping function
        a = np.empty(structure.shape)
        for idx in np.ndindex(structure.shape):
            a[idx] = nested_map(fun, structure[idx])
        return a
    return fun(structure)


def nested_depth(structure: Any) -> int:
    """
    Calculate the maximum depth of a nested sequence.

    :param structure: nested sequence. The containing data structures are currently restricted to lists and tuples,
     because allowing any sequence would also result in traversing strings for example.
     If a single value is passed, the return is 0
    :return: maximum depth
    """
    if isinstance(structure, (list, tuple, set)):
        if structure:
            # if the list contains elements
            return 1 + max((nested_depth(l_) for l_ in structure))
        return 1
    return 0


def fill_params(template: Union[list, tuple, Mapping, np.ndarray], data: Union[pd.Series, Mapping]) \
        -> Union[list, tuple, Mapping, np.ndarray]:
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
    return nested_map(lambda k: data.get(k, k), template)


def flatten_together(structure: List[Union[List, Any]], values: Union[Any, List[Union[List, Any]]]):
    """
    Flattens and fills a list of values relative to the groupings provided by the structure parameter.
    The explicit values in the structure parameter are ignored. Only the nesting structure is of importance.
    If a single value is provided it is simply repeated as often as the structure has values

    e.g. when called with :code:`[[0, 0], [0, 0]]` and :code:`[[0, None], 4]` it will detect the grouping
    and return :code:`[0, None, 4, 4]`


    :param structure: nested list used as a template
    :param values: values matched to the list
    :return: flattened and filled list of values
    """
    if not isinstance(structure, list):
        if isinstance(values, list):
            raise ValueError('There where to many nestings in the values')
        return values
    if not isinstance(values, list):
        values = [values]
    if len(structure) < len(values):
        return flatten_together(collapse(structure, base_type=tuple, levels=1), values)
    elif len(structure) > len(values):
        # if structure has more elements we need to repeat value elements
        values = values * (len(structure) // len(values))
        if len(structure) != len(values):
            raise ValueError('stuff does not match up')
    return list(collapse([flatten_together(s, v) for s, v in zip(structure, values)], base_type=tuple))
