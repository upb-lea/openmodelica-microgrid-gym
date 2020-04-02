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
        pass

    @property
    def cols(self):
        return flatten(self._structured_cols)

    @cols.setter
    def cols(self, val):
        self._structured_cols = val

    def structured_cols(self, remaining_level=1):
        return flatten(self._structured_cols, remaining_level)


class SingleHistory(EmptyHistory):
    """
    Full history that stores all data
    """

    # TODO improve inheritancy, DRY out code
    def append(self, values, cols=None):
        newcols = []
        if isinstance(values, pd.DataFrame):
            if cols is not None:
                raise ValueError('providing columns with DataFrames is not supported. '
                                 'Maybe you want to do ".append(df[cols])" instead.')
            self.df = values
            if set(self.cols) is not set(self.df.columns):
                # if colums have been added, we append them to the cols
                newcols = [col for col in self.df.columns if col not in set(self.cols)]
        elif isinstance(values, Sequence):
            self.df = pd.DataFrame([values], columns=cols or self.cols)
            if cols is not None:
                newcols = [col for col in cols if col not in set(self.cols)]
        else:
            raise ValueError('"values" must be a sequence or DataFrame')

        self.cols += newcols


class FullHistory(EmptyHistory):
    """
    Full history that stores all data
    """

    def append(self, values, cols=None):
        newcols = []
        if isinstance(values, pd.DataFrame):
            if cols is not None:
                raise ValueError('providing columns with DataFrames is not supported. '
                                 'Maybe you want to do ".append(df[cols])" instead.')
            self.df = self.df.append(values, ignore_index=True)
            if set(self.cols) is not set(self.df.columns):
                # if colums have been added, we append them to the cols
                newcols = [col for col in self.df.columns if col not in set(self.cols)]
        elif isinstance(values, Sequence):
            self.df = self.df.append(pd.DataFrame([values], columns=cols or self.cols), ignore_index=True)
            if cols is not None:
                newcols = [col for col in cols if col not in set(self.cols)]
        else:
            raise ValueError('"values" must be a sequence or DataFrame')

        self.cols += newcols
