from typing import Sequence

import pandas as pd


class EmptyHistory:
    """
    Dummy history for recording data in the environment
    This class will not actually store any data
    """

    def __init__(self, cols=None):
        self.cols = cols
        self.df: pd.DataFrame

    def reset(self):
        self.df = pd.DataFrame([], columns=self.cols)

    def append(self, values, cols=None):
        pass

    def __str__(self):
        return self.df.__str__()

    def __getitem__(self, item):
        return self.df[item]


class FullHistory(EmptyHistory):
    """
    Full history that stores all data
    """

    def __init__(self, cols=None):
        super().__init__(cols)

    def append(self, values, cols=None):
        newcols = []
        if isinstance(values, pd.DataFrame):
            if cols is not None:
                raise ValueError('providing columns with DataFrames is not supported. '
                                 'Maybe you want to do ".append(df[cols])" instead.')
            self.df = self.df.append(values)
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
