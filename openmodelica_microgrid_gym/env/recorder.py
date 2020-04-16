from typing import Sequence, List, Optional, Union

import pandas as pd
import numpy as np

from openmodelica_microgrid_gym.common.itertools_ import flatten


class EmptyHistory:
    """
    Dummy history for recording data in the environment
    This class will not actually store any data
    """

    def __init__(self, cols: List[Union[List, str]] = None):
        """

        :param cols: nested lists of strings providing column names and hierarchical structure
        """
        if cols is None:
            cols = []
        self._structured_cols = cols

        self.df: pd.DataFrame = pd.DataFrame([])
        """internal pd.DataFrame storing the history data"""

    def reset(self):
        """
        Removes all data, but keeps the columns.
        """
        self.df = pd.DataFrame([], columns=self.cols)

    def append(self, values: Union[pd.Series, Sequence], cols: Optional[List[str]] = None):
        """
        Add new data sample to the history. The History class will determine how the data is updated

        :param values: data. If provided as a pd.Series, the columns of the pd.Series are used to map the columns
        :param cols: only supported if values is a np.array. It will determine in which columns the data should be added
        """
        if isinstance(values, pd.Series):
            if cols is not None:
                raise ValueError('providing columns with Series is not supported. '
                                 'Maybe you want to do ".append(series[idx])    " instead.')
            self.df = self._append(values.to_frame().T)
        elif isinstance(values, (Sequence, np.ndarray)):
            self.df = self._append(pd.DataFrame([values], columns=cols or self.cols))
        else:
            raise ValueError('"values" must be a sequence or pd.Series')

    def update(self, values: Union[pd.Series, Sequence], cols: Optional[List[str]] = None):
        """
        Updates/Overrides/Extends the last entry of the history.

        :param values: data. If provided as a pd.Series, the columns of the pd.Series are used to map the columns
        :param cols: only supported if values is a np.array. It will determine in which columns the data should be added
         """
        if isinstance(values, pd.Series):
            if cols is not None:
                raise ValueError('providing columns with Series is not supported. '
                                 'Maybe you want to do ".append(series[idx])" instead.')
            self.df.iloc[-1] = values.to_frame().T
        elif isinstance(values, (Sequence, np.ndarray)):
            self.df.iloc[-1] = pd.DataFrame([values], columns=cols or self.cols).iloc[-1]
        else:
            raise ValueError('"values" must be a sequence or Series')

    @property
    def cols(self) -> List[str]:
        """
        Columns of the History

        :return: Flat list of the columns
        """
        return flatten(self._structured_cols)

    @cols.setter
    def cols(self, val: List[Union[List, str]]):
        """
        Columns of the History

        :param val: Nested list of columns as string
        """
        self._structured_cols = val

    def structured_cols(self, remaining_level: Optional[int] = 1) -> List[Union[List, str]]:
        """
        Get columns with the specified amount of levels retained

        :param remaining_level: number of levels to retain
        """
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
