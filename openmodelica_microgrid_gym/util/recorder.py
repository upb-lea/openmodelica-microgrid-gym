from typing import Sequence, List, Optional, Union

import pandas as pd

from openmodelica_microgrid_gym.util.itertools_ import flatten


class StructuredMapping:
    def __init__(self, cols: List[Union[List, str]] = None, data=None):
        """

        :param cols: nested lists of strings providing column names and hierarchical structure
        """
        if cols is None:
            cols = []
        self.cols = cols

        self._data = data
        """internal field storing the history data"""

    @property
    def cols(self) -> List[str]:
        """
        Columns of the History

        :return: Flat list of the columns
        """
        return self._cols

    @cols.setter
    def cols(self, val: List[Union[List, str]]):
        """
        Columns of the History

        :param val: Nested list of columns as string
        """
        self._structured_cols = val
        self._cols = flatten(val)

    @property
    def data(self):
        return self._data

    @property
    def df(self):
        # executing this conditionally only if _data is not a df is actually slower!!!
        return pd.DataFrame([self._data], columns=self.cols)

    def structured_cols(self, remaining_level: Optional[int] = 1) -> List[Union[List, str]]:
        """
        Get columns with the specified amount of levels retained

        :param remaining_level: number of levels to retain
        """
        return flatten(self._structured_cols, remaining_level)


class EmptyHistory(StructuredMapping):
    """
    Dummy history for recording data in the environment
    This class will not actually store any data
    """

    def reset(self):
        """
        Removes all data, but keeps the columns.
        """
        self._data = []

    def pop(self):
        """
        pop last item
        :return:
        """
        pass

    def append(self, values: Sequence):
        """
        Add new data sample to the history. The History class will determine how the data is updated

        :param values: sequence of data entries
        """
        pass

    def last(self):
        return self.df.tail(1).squeeze()

    def __getitem__(self, item):
        return self.df[item]


class SingleHistory(EmptyHistory):
    """
    Single history that stores only the last added value
    """

    def reset(self):
        self._data = None

    def pop(self):
        val = self.last()
        self.reset()
        return val

    def append(self, values: Sequence):
        self._data = values

    def last(self):
        return self._data


class FullHistory(EmptyHistory):
    """
    Full history that stores all data
    """

    def reset(self):
        self._data = []

    def pop(self):
        return self._data.pop(-1)

    def append(self, values: Sequence):
        self._data.append(list(values))

    def last(self):
        return self._data[-1]

    @property
    def df(self):
        # executing this conditionally only if _data is not a df is actually slower!!!
        return pd.DataFrame(self._data, columns=self.cols)
