import sys
from itertools import chain
from typing import List, Optional

import numpy as np
import yaml

from openmodelica_microgrid_gym.aux_ctl import PLL, PLLParams


class Network:
    def __init__(self, ts, v_nom, freq_nom=50):
        self.ts = ts
        self.v_nom = v_nom
        self.freq_nom = freq_nom
        self.components = []  # type: List[Component]

    @classmethod
    def load(cls, configurl='net.yaml'):
        """
        Initialize object from config file

        :param configurl:
        :return:
        """
        data = yaml.safe_load(open(configurl))
        components = data['components']
        del data['components']
        self = cls(**data)

        for name, component in components.items():
            # resolve class from 'cls' argument
            comp_cls = component['cls']
            del component['cls']
            comp_cls = getattr(sys.modules[__name__], comp_cls)

            # rename keys (because 'in' is a reserved keyword)
            if 'in' in component:
                component['in_vars'] = component.pop('in')
            if 'out' in component:
                component['out_vars'] = component.pop('out')

            # instanciate component class
            try:
                self.components.append(comp_cls(net=self, **component))
            except AttributeError as e:
                raise AttributeError(f'{e!s}, please validate {configurl}')

        return self

    def augment_actions(self, actions):
        """
        Allows the network to add additional parameters like changing loads to the simulation

        :param actions:
        :return: mapping of additional actions and list of actions.
        """
        # TODO get params from components
        return {}, actions

    def augment_state(self, state: np.ndarray) -> np.ndarray:
        """
        Allows the network to provide additional output variables in order to provide measurements and reference
        information the RL agent needs to understand its rewards
        :param state:
        :return:
        """
        # TODO calculate additional outputs
        return np.hstack([comp.normalized_outputs(state) for comp in self.components])






    def _statekeys(self, keys):
        for comp in self.components:
            comp.set_outidx(keys)


    def in_keys(self):
        return [comp.inkeys(onlysrc) for comp in self.components]

    def out_keys(self, onlysrc=False):
        return [comp.outkeys(onlysrc) for comp in self.components]


class Component:
    def __init__(self, net: Network, id, in_vars=None, out_vars=None):
        self.net = net
        self.id = id
        for attr in chain.from_iterable((f.keys() for f in filter(None, (in_vars, out_vars)))):
            if not hasattr(self, attr):
                raise AttributeError(f'{self.__class__} no such attribute: {attr}')
        self.in_vars = in_vars
        self.in_idx = None  # type: Optional[dict]

        self.new_outputs
        self.out_vars = out_vars
        self.out_idx = None  # type: Optional[dict]

    def inputs(self):
        pass

    def outputs(self, state):
        self.fill_tmpl(state)
        return np.hstack([getattr(self, attr) for attr in self.out_idx.keys()])

    def inkeys(self, onlysrc=False):
        if onlysrc:
            if self.in_vars:
                return [['.'.join([self.id, val]) for val in vals] for attr, vals in self.in_vars.items()]
            return []

    def outkeys(self, onlysrc=False):
        if onlysrc:
            if self.out_vars:
                return [['.'.join([self.id, val]) for val in vals] for attr, vals in self.out_vars.items()]
            return []
        else:
            return [['.'.join([self.id, attr, str(i)]) for i, _ in enumerate(vals)] for attr, vals in
                    self.out_vars.items()]

    def fill_tmpl(self, state):
        if self.out_idx is None:
            raise ValueError('call set_tmplidx before fill_tmpl. the keys must be converted to indices for efficiency')
        for attr, idxs in self.out_idx.items():
            # set object variables to the respective state variables
            if hasattr(self, attr):
                setattr(self, attr, state[idxs])
            else:
                raise AttributeError(f'{self.__class__} has no such attribute: {attr}')

    def set_outidx(self, statekeys):
        # this is mainly for performance reasons
        keyidx = {v: i for i, v in enumerate(statekeys)}
        self.out_idx = {}
        try:
            for var, keys in self.out_vars.items():
                # lookup index in the whole state keys
                self.out_idx[var] = [keyidx[f'{self.id}.{key}'] for key in keys]
        except KeyError as e:
            raise KeyError(f'the output variable {e!s} is not provided by your state keys')

    def normalized_outputs(self, state):
        return self.outputs(state)


class Inverter(Component):
    def __init__(self, u=None, i=None, v=None, i_nom=20, i_lim=30, v_DC=1000, iref=(0, 0, 0), **kwargs):
        self.u = u
        self.v = v
        self.i = i
        self.i_nom = i_nom
        self.i_lim = i_lim
        self.v_DC = v_DC
        self.i_ref = iref
        super().__init__(**kwargs)

    def outputs(self, state):
        # calculate ABC conversion and add references to outputs
        return super().outputs(state)


class SlaveInverter(Inverter):
    def __init__(self, pll=None, **kwargs):
        super().__init__(**kwargs)
        if pll is None:
            pll = {}
        else:
            # additional shorthand to be able to use p and i instead of kP and kI
            if 'p' in pll:
                pll['kP'] = pll.pop('p')
            if 'i' in pll:
                pll['kI'] = pll.pop('i')
        # default pll params and new ones
        self.pll = PLL(PLLParams(**{**dict(kP=10, kI=200, f_nom=self.net.freq_nom), **pll}), self.net.ts)


class MasterInverter(Inverter):
    def __init__(self, vref=(1, 0, 0), **kwargs):
        super().__init__(**kwargs)

    def outputs(self, state):
        out = super().outputs(state)


class Load(Component):
    def __init__(self, i=None, **kwargs):
        self.i = i
        super().__init__(**kwargs)

    def inputs(self):
        # perhaps provide modelparams that set resistance value
        return super().inputs()


if __name__ == '__main__':
    import numpy as np

    # env = ModelicaEnv()

    net = Network.load('net.yaml')
    # net.statekeys(env.model_output_names)
    net.statekeys([f'lc1.capacitor{k}.v' for k in '123'] + [f'lcl1.capacitor{k}.v' for k in '123']
                  + [f'lc1.inductor{k}.i' for k in '123'] + [f'lcl1.inductor{k}.i' for k in '123']
                  + [f'rl1.inductor{k}.i' for k in '123'])
    print(net.augment(np.array([2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4])))
