from typing import List, Union, Callable, Optional

from matplotlib.figure import Figure
from more_itertools import collapse

from openmodelica_microgrid_gym.util.itertools_ import flatten_together


class PlotTmpl:
    def __init__(self, vars: List[Union[List, str]], callback: Optional[Callable[[Figure], None]] = None, **kwargs):

        self.vars = list(collapse(vars))
        self.callback = callback

        # set colors None if not provided
        if not (colorkey := ({'c', 'color'} & set(kwargs.keys()))):
            kwargs['c'] = None
            colorkey = 'c'
        elif len(colorkey) > 1:
            raise ValueError(f'Multiple color parameters provided "{colorkey}"')
        else:
            colorkey = colorkey.pop()

        args = dict()
        for k, v in dict(kwargs).items():
            args[k] = flatten_together(vars, v)

        # apply to a group only if all color values are none inside that group
        if colorkey:
            # if all elements in the variables are lists and they are all of equal length
            if len(lengths := set([isinstance(l, list) and len(l) for l in vars])) == 1:
                # set contains either the length of all lists or false if all values where non-list values
                if length := lengths.pop():
                    for groups in range(len(vars)):
                        for i in range(length):
                            if args[colorkey][length * groups + i] is None:
                                args[colorkey][length * groups + i] = 'C' + str(i + 1)
            else:
                ## all elements are single values
                for i, c in enumerate(args[colorkey]):
                    if c is None:
                        args[colorkey][i] = 'C' + str(i + 1)

        # merge parameters to the variables for indexing access
        self.kwargs = []
        for i, _ in enumerate(self.vars):
            args_ = dict()
            for k, arg in args.items():
                v = arg[i]
                if v is not None:
                    args_[k] = v
            self.kwargs.append(args_)

    def __iter__(self):
        self.i = -1
        return self

    def __next__(self):
        try:
            self.i += 1
            return self.vars[self.i], self.kwargs[self.i]
        except IndexError:
            raise StopIteration

    def __getitem__(self, item):
        return self.vars[item], self.kwargs[item]
