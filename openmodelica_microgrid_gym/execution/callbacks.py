from abc import ABC


class Callback(ABC):
    def reset(self):
        pass

    def __call__(self, *args, **kwargs):
        pass



