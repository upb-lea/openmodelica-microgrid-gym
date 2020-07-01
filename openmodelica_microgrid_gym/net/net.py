import sys
from itertools import chain
from typing import List, Optional

import numpy as np
import yaml
from more_itertools import flatten, collapse
import numexpr as ne

from openmodelica_microgrid_gym.aux_ctl import PLL, PLLParams


class Network:
    def __init__(self, ts, v_nom, freq_nom=50):
        self.ts = ts
        self.v_nom = v_nom
        self.freq_nom = freq_nom

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

        components_obj = []
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
                components_obj.append(comp_cls(net=self, **component))
            except AttributeError as e:
                raise AttributeError(f'{e!s}, please validate {configurl}')
        self.components = components_obj

        return self

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

    def augment(self, state: np.ndarray) -> np.ndarray:
        """
        Allows the network to provide additional output variables in order to provide measurements and reference
        information the RL agent needs to understand its rewards
        :param state:
        :return:
        """
        return np.hstack([comp.augment(state) for comp in self.components])

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


class Component:
    def __init__(self, net: Network, id=None, in_vars=None, out_vars=None, out_calc=None):
        """

        :param net:
        :param id:
        :param in_vars:
        :param out_vars:
        :param out_calc: mapping from attr name to
        """
        self.net = net
        self.id = id
        for attr in chain.from_iterable((f.keys() for f in filter(None, (in_vars, out_vars)))):
            if not hasattr(self, attr):
                raise AttributeError(f'{self.__class__} no such attribute: {attr}')
        self.in_vars = in_vars
        self.in_idx = None  # type: Optional[dict]

        self.out_calc = out_calc or {}
        self.out_vars = out_vars
        self.out_idx = None  # type: Optional[dict]

    def params(self, actions):
        """
        Calculate additional environment parameters
        :param actions:
        :return: mapping
        """
        return {}

    def get_in_vars(self):
        """
        list of input variable names of this component
        """
        if self.in_vars:
            return [[self._prefix_var(val) for val in vals] for attr, vals in self.in_vars.items()]
        return []

    def get_out_vars(self, with_aug=False):
        r = []
        if not with_aug:
            if self.out_vars:
                r = [[self._prefix_var(val) for val in vals] for attr, vals in self.out_vars.items()]
            return r
        else:
            return r + [[self._prefix_var([attr, str(i)]) for i in range(n)] for attr, n in self.out_calc.items()]

    def fill_tmpl(self, state):
        if self.out_idx is None:
            raise ValueError('call set_tmplidx before fill_tmpl. the keys must be converted to indices for efficiency')
        for attr, idxs in self.out_idx.items():
            # set object variables to the respective state variables
            if hasattr(self, attr):
                setattr(self, attr, state[idxs])
            else:
                raise AttributeError(f'{self.__class__} has no such attribute: {attr}')

    def set_outidx(self, keys):
        # this is mainly for performance reasons
        keyidx = {v: i for i, v in enumerate(keys)}
        self.out_idx = {}
        try:
            for var, keys in self.out_vars.items():
                # lookup index in the whole state keys
                self.out_idx[var] = [keyidx[self._prefix_var(key)] for key in keys]
        except KeyError as e:
            raise KeyError(f'the output variable {e!s} is not provided by your state keys')

    def _prefix_var(self, strs):
        if isinstance(strs, str):
            strs = [strs]
        if self.id is not None:
            strs = [self.id] + strs
        return '.'.join(strs)

    def calculate(self):
        """
        will write internal variables it is called after all internal variables are set
        The return value must be a dictionary whose keys match the keys of self.out_calc and whose values are of the length of outcalcs values

        set(self.out_calc.keys()) == set(return)
        all([len(v) == self.out_calc[k] for k,v in return.items()])
        :return:
        """
        return {}

    def augment(self, state):
        self.fill_tmpl(state)
        calc_data = self.calculate()
        # TODO normalize outputs1
        attr = ''
        try:
            new_vals = []
            for attr, n in self.out_calc.items():
                for i in range(n):
                    new_vals.append(calc_data[attr][i])
            return np.hstack([getattr(self, attr) for attr in self.out_idx.keys()] + new_vals)
        except KeyError as e:
            raise ValueError(
                f'{self.__class__} missing return key: {e!s}. did you forget to set it in the calculate method?')
        except IndexError as e:
            raise ValueError(f'{self.__class__}.calculate()[{attr}] has the wrong number of values')


class Inverter(Component):
    def __init__(self, u=None, i=None, v=None, i_nom=20, i_lim=30, v_DC=1000, iref=(0, 0, 0), **kwargs):
        self.u = u
        self.v = v
        self.i = i
        self.i_nom = i_nom
        self.i_lim = i_lim
        self.v_DC = v_DC
        self.i_ref = iref
        super().__init__(**{'out_calc': dict(iref=3), **kwargs})

    def calculate(self):
        # TODO: calculate ABC conversion and add references to outputs
        #        self.eval = ne.evaluate('self.u/self.i')
        return dict(iref=[1, 2, 4])


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
        self.vref = vref
        super().__init__(out_calc=dict(iref=3, vref=3), **kwargs)

    def calculate(self):
        # TODO: calculate ABC conversion and add references to outputs

        return {**super().calculate(), **dict(vref=[4, 3, 3])}


class Load(Component):
    def __init__(self, i=None, **kwargs):
        self.i = i
        super().__init__(**kwargs)

    def params(self, actions):
        # TODO: perhaps provide modelparams that set resistance value
        return super().params(actions)


if __name__ == '__main__':
    import numpy as np

    # env = ModelicaEnv()

    net = Network.load('net.yaml')
    # net.keys(env.model_output_names)
    net.statekeys([f'lc1.capacitor{k}.v' for k in '123'] + [f'lcl1.capacitor{k}.v' for k in '123']
                  + [f'lc1.inductor{k}.i' for k in '123'] + [f'lcl1.inductor{k}.i' for k in '123']
                  + [f'rl1.inductor{k}.i' for k in '123'])
    print(net.augment(np.array([2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 4, 4, 4])))
