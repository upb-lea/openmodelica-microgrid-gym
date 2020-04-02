from typing import Sequence


class MutableFloat():
    def __init__(self, f: float):
        self._f = f

    def __float__(self):
        return float(self.val)

    def __repr__(self):
        return f'{self.__class__.__name__}({float(self)})'

    @property
    def val(self):
        return self._f

    @val.setter
    def val(self, v):
        self._f = v


class MutableParams:
    def __init__(self, vars: Sequence[MutableFloat]):
        self.vars = vars
        self.defaults = [float(v) for v in vars]

    def reset(self):
        for var, default in zip(self.vars, self.defaults):
            var.val = default

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            for var, val in zip(self.vars[key], value):
                var.val = val
        else:
            self.vars[key].val = value

    def __getitem__(self, item):
        if isinstance(item, slice):
            return [float(v) for v in self.vars[item]]
        return float(self.vars[item])

    def __repr__(self):
        return str(list(self.vars))
