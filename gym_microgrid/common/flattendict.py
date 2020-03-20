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
"""
from typing import Sequence, Callable

import pandas as pd
from more_itertools import collapse


def flatten(data, remaining_levels=0):
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
    return list(collapse(data, levels=depth - remaining_levels - 1))


def nested_map(l: Sequence, fun: Callable):
    if isinstance(l, list):
        return [nested_map(l_, fun) for l_ in l]
    return fun(l)


def nested_depth(l: Sequence) -> int:
    if isinstance(l, list):
        if l:
            # if the list contains elements
            return max((nested_depth(l_) for l_ in l)) + 1
        return 1
    return 0
