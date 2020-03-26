"""
df + datastructure of keys -> datastructure of values
supported datastructures: Dict[List[Union(np.ndarray,float]]]

{'ctl1':[np.array(['inductor1.i', 'inductor2.i', 'inductor3.i']),
         np.array(['capacitor1.v', 'capacitor2.v', 'capacitor3.v'])],


"""
from operator import getitem

import pandas as pd


def fill_params(df: pd.DataFrame, template):
    d = df.to_dict()
    getitem
