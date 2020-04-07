from typing import Sequence

import pandas as pd

from gym_microgrid.common.itertools_ import flatten


class EmptyHistory:
    """
    Dummy history for recording data in the environment
    This class will not actually store any data
    """

    def __init__(self, cols=None):
        if cols is None:
            cols = []
        self._structured_cols = cols

        self.df: pd.DataFrame

    def reset(self):
        self.df = pd.DataFrame([], columns=self.cols)

    def append(self, values, cols=None):
        if isinstance(values, pd.DataFrame):
            if cols is not None:
                raise ValueError('providing columns with DataFrames is not supported. '
                                 'Maybe you want to do ".append(df[cols])" instead.')
            self.df = self._append(values)
        elif isinstance(values, Sequence):
            self.df = self._append(pd.DataFrame([values], columns=cols or self.cols))
        else:
            raise ValueError('"values" must be a sequence or DataFrame')

    def update(self, values, cols=None):
        if isinstance(values, pd.DataFrame):
            if cols is not None:
                raise ValueError('providing columns with DataFrames is not supported. '
                                 'Maybe you want to do ".append(df[cols])" instead.')
            self.df.iloc[-1] = values.iloc[-1]
        elif isinstance(values, Sequence):
            self.df.iloc[-1] = pd.DataFrame([values], columns=cols or self.cols).iloc[-1]
        else:
            raise ValueError('"values" must be a sequence or DataFrame')

    @property
    def cols(self):
        return flatten(self._structured_cols)

    @cols.setter
    def cols(self, val):
        self._structured_cols = val

    def structured_cols(self, remaining_level=1):
        return flatten(self._structured_cols, remaining_level)

    def __getitem__(self, item):
        return self.df[item]

    def _append(self, values):
        pass


class SingleHistory(EmptyHistory):
    """
    Single history that stores only the last added value
    """

    def _append(self, new):
        return new


class FullHistory(EmptyHistory):
    """
    Full history that stores all data
    """

    def _append(self, new):
        return self.df.append(new, ignore_index=True)
