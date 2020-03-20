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

import pandas as pd
import numpy as np
from more_itertools import collapse


def flatten(data, remaining_levels=0):
    # collapse outer dicts
    if isinstance(data, dict):
        # flatten all the dicts
        df = pd.json_normalize(data)
        data = df.to_dict(orient='records')[0]
        # move the key into the lists
        for k, v in data.items():
            f = np.vectorize(lambda s, t: '.'.join([t, s]))
            data[k] = f(np.array(v), k).tolist()
        data = list(data.values())
    # count levels and collapse to keep the levels as needed
    depth = len(np.array(data).shape)
    return list(collapse(data, levels=depth - remaining_levels - 1))
