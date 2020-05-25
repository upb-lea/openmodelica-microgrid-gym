import numpy as np
import pandas as pd

from openmodelica_microgrid_gym.util import FullHistory


def test__append():
    rec = FullHistory(['a b c'.split()])
    rec.reset()
    rec.append(np.array([1, 2, 3]))
    rec.append([3, 3, 3])

    assert rec.df.equals(pd.DataFrame([dict(a=1, b=2, c=3), dict(a=3, b=3, c=3)]))
