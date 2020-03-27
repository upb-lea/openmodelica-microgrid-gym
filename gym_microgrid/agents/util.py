from dataclasses import dataclass


@dataclass
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
