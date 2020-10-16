from importlib import import_module

import numexpr as ne
import numpy as np
import yaml
from more_itertools import collapse, flatten


class Network:
    def __init__(self, ts, v_nom, freq_nom=50):
        self.ts = float(ts)
        self.v_nom = ne.evaluate(str(v_nom))
        self.freq_nom = freq_nom

    @staticmethod
    def _validate_load_data(data):
        # validate that inputs are disjoined
        components_with_inputs = [component['in'].values() for component in data['components'].values() if
                                  'in' in component]
        inputs = list(flatten(components_with_inputs))
        if sum(map(len, inputs)) != len(set().union(*inputs)):
            # all inputs are pairwise disjoint if the total number of inputs is the same as the number of elements in the union
            raise ValueError('The inputs of the components should be disjoined')

        return True

    @classmethod
    def load(cls, configurl='net.yaml'):
        """
        Initialize object from config file

        :param configurl:
        :return:
        """
        data = yaml.safe_load(open(configurl))
        if not cls._validate_load_data(data):
            raise ValueError(f'loading {configurl} failed due to validation')
        components = data['components']
        del data['components']
        self = cls(**data)

        components_obj = []
        for name, component in components.items():
            # resolve class from 'cls' argument
            comp_cls = component['cls']
            del component['cls']
            comp_cls = getattr(import_module('..components', __name__), comp_cls)

            # rename keys (because 'in' is a reserved keyword)
            if 'in' in component:
                component['in_vars'] = component.pop('in')
            if 'out' in component:
                component['out_vars'] = component.pop('out')

            # instanciate component class
            try:
                components_obj.append(comp_cls(net=self, **component))
            except AttributeError as e:
                raise AttributeError(f'{e!s}, please validate {configurl}')
        self.components = components_obj

        return self

    def risk(self) -> float:
        #TODO call risk of all components
        return 0

    def reset(self):
        for comp in self.components:
            comp.reset()

    def params(self, actions):
        """
        Allows the network to add additional parameters like changing loads to the simulation

        :param actions:
        :return: mapping of additional actions and list of actions.
        """
        d = {}
        for comp in self.components:
            params = comp.params(actions)
            d.update(params)
        return d

    def augment(self, state: np.ndarray, normalize=True) -> np.ndarray:
        """
        Allows the network to provide additional output variables in order to provide measurements and reference
        information the RL agent needs to understand its rewards
        :param state:
        :return:
        """
        return np.hstack([comp.augment(state, normalize) for comp in self.components])

    def in_vars(self):
        return list(collapse([comp.get_in_vars() for comp in self.components]))

    def out_vars(self, with_aug=True, flattened=True):
        r = [comp.get_out_vars(with_aug) for comp in self.components]
        if flattened:
            return list(collapse(r))
        return r

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, val):
        self._components = val
        keys = self.out_vars(with_aug=False, flattened=True)
        for comp in self.components:
            comp.set_outidx(keys)
