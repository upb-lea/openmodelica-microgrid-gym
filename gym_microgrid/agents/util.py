from dataclasses import dataclass


@dataclass
class MutableFloat():
    def __init__(self, f: float):
        self._f = f

    def __float__(self):
        return float(self.val)

    @property
    def val(self):
        return self._f

    @val.setter
    def val(self, v):
        self._f = v
