import pandas as pd


class History:
    def __init__(self, cols):
        self.cols = cols
        self.df = None

    def reset(self):
        self.df = pd.DataFrame([], columns=self.cols)

    def append(self, values):
        pass

    def __str__(self):
        return self.df.__str__()


class FullHistory(History):
    def __init__(self, cols):
        super().__init__(cols)

    def append(self, values):
        self.df = self.df.append(pd.DataFrame([values], columns=self.cols), ignore_index=True)
