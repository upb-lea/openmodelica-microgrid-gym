from typing import List, Union, Callable

from matplotlib.figure import Figure
from more_itertools import collapse


class PlotTmpl:
    def __init__(self, vars: List[Union[List, str]], callback: Callable[[Figure], None], **kwargs):
        # match colorings if there are n lists with equal length

        self.vars = collapse(vars)
        self.callback = callback

        self.kwargs

        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        self.i += 1
        return self.vars[self.i], self.kwargs[self.i]
