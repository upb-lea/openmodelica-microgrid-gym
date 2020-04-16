from openmodelica_microgrid_gym.env import FullHistory
import pandas as pd
import numpy as np


def test__append():
    rec = FullHistory(['a b c'.split()])
    rec.append(np.array([1, 2, 3]))
    rec.append([3, 3, 3])

    assert rec.df.equals(pd.DataFrame([dict(a=1, b=2, c=3), dict(a=3, b=3, c=3)]))
