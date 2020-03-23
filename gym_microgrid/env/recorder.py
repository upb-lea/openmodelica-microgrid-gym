import pandas as pd


class EmptyHistory:
    """
    Dummy history for recording data in the environment
    This class will not actually store any data
    """

    def __init__(self, cols=None):
        self.cols = cols
        self.df = None

    def reset(self):
        self.df = pd.DataFrame([], columns=self.cols)

    def append(self, values):
        pass

    def __str__(self):
        return self.df.__str__()


class FullHistory(EmptyHistory):
    """
    Full history that stores all data
    """

    def __init__(self, cols=None):
        super().__init__(cols)

    def append(self, values):
        self.df = self.df.append(pd.DataFrame([values], columns=self.cols), ignore_index=True)
