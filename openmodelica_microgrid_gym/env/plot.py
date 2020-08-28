from typing import List, Union, Callable, Optional

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from more_itertools import collapse

from openmodelica_microgrid_gym.util import flatten_together


class PlotTmpl:
    def __init__(self, variables: List[Union[List, str]], callback: Optional[Callable[[Figure], None]] = None,
                 **kwargs):
        """
        Provides an iterable of variables and plot parameters like ('induction1', {'color':'green', 'style': '--'}).
        It contains logic to automatically match up the variables and provided kwargs to allow for a simple syntax.

        e.g. when called with [['a','b'],['c','d']] and style = [['.', None],'--'] it will detect the grouping and apply
        the dotted style to 'a' and the dashed style to 'c' and 'd'

        :param vars: nested list of strings.
         Each string represents a variable of the FMU or a measurement that should be plotted by the environment.
        :param callback: if provided, it is executed after the plot is finished.
         Will get the generated figure as parameter to allow further modifications.
        :param kwargs: those arguments are merged (see omg.util.flatten_together) with the variables
         and than provided to the pd.DataFrame.plot(·) function
        """

        self.vars = list(collapse(variables))
        self._callback = callback

        # set colors None if not provided
        colorkey = ({'c', 'color'} & set(kwargs.keys()))
        if not colorkey:
            kwargs['c'] = None
            colorkey = 'c'
        elif len(colorkey) > 1:
            raise ValueError(f'Multiple color parameters provided "{colorkey}"')
        else:
            colorkey = colorkey.pop()

        args = dict()
        for k, v in dict(kwargs).items():
            args[k] = flatten_together(variables, v)

        # apply to a group only if all color values are none inside that group
        if colorkey:
            # if all elements in the variables are lists and they are all of equal length
            lengths = set([isinstance(l, list) and len(l) for l in variables])
            if len(lengths) == 1:
                # set contains either the length of all lists or false if all values where non-list values
                length = lengths.pop()
                if length:
                    for groups in range(len(variables)):
                        for i in range(length):
                            if args[colorkey][length * groups + i] is None:
                                args[colorkey][length * groups + i] = 'C' + str(i + 1)
            else:
                # all elements are single values
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

    def callback(self, fig: Figure):
        """
        Will be called in the ModelicaEnv.render() once all plotting is finished.
        This function enables the user to specify more modifications to apply to the figure.
        The function will call the callable passed in the constructor.
        Additionally the figure is plotted by this function.

        :param fig: Finished figure that is one might want to modify.
        :return:
        """
        if self._callback is not None:
            self._callback(fig)
        else:
            plt.show()


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
