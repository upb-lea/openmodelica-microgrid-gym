import sys
from itertools import chain
from typing import List, Optional

import numpy as np
import yaml
from more_itertools import flatten, collapse
import numexpr as ne

from openmodelica_microgrid_gym.aux_ctl import PLL, PLLParams, dq0_to_abc, inst_power, inst_reactive, DDS, DroopParams, \
    DroopController, InverseDroopController, InverseDroopParams


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

    def reset(self):
        pass

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
        if self.out_vars:
            r = [[self._prefix_var(val) for val in vals] for attr, vals in self.out_vars.items()]

        if not with_aug:
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
        if self.id is not None and strs[0].startswith('.'):
            # this is a complete identifier like 'lc1.inductor1.i' that should not be modified:
            # first string minus its prefix '.' and the remaining strs
            strs = [self.id] + [strs[0][1:]] + strs[1:]
        return '.'.join(strs)

    def calculate(self):
        """
        will write internal variables it is called after all internal variables are set
        The return value must be a dictionary whose keys match the keys of self.out_calc and whose values are of the length of outcalcs values

        set(self.out_calc.keys()) == set(return)
        all([len(v) == self.out_calc[k] for k,v in return.items()])
        :return:
        """
        return dict(iref=[.1, 22, 4], vref=[])

    def normalize(self, calc_data):
        pass

    def augment(self, state, normalize=True):
        self.fill_tmpl(state)
        calc_data = self.calculate()

        if normalize:
            self.normalize(calc_data)
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
    def __init__(self, u=None, i=None, v=None, i_nom=20, i_lim=30, v_lim=600, v_DC=1000, i_ref=(0, 0, 0), **kwargs):
        self.u = u
        self.v = v
        self.i = i
        self.i_nom = i_nom
        self.i_lim = i_lim
        self.v_lim = v_lim
        self.v_DC = v_DC
        self.i_ref = i_ref
        super().__init__(**{'out_calc': dict(i_ref=3), **kwargs})

    def normalize(self, calc_data):
        self.i /= self.i_lim
        self.v /= self.v_lim
        calc_data['i_ref'] /= self.i_lim

    def params(self, actions):
        return {**super().params(actions), **{self._prefix_var(['.v_DC']): self.v_DC}}


class SlaveInverter(Inverter):
    def __init__(self, pll=None, pdroop=None, qdroop=None, **kwargs):
        super().__init__(**kwargs)

        pdroop = {**dict(gain=40000.0), **(pdroop or {})}
        qdroop = {**dict(gain=50.0), **(qdroop or {})}
        pll = {**dict(kP=10, kI=200), **(pll or {})}

        self.pdroop_ctl = InverseDroopController(
            InverseDroopParams(tau=self.net.ts, nom_value=self.net.freq_nom, **pdroop), self.net.ts)
        self.qdroop_ctl = InverseDroopController(
            InverseDroopParams(tau=self.net.ts, nom_value=self.net.v_nom, **qdroop), self.net.ts)
        # default pll params and new ones
        self.pll = PLL(PLLParams(f_nom=self.net.freq_nom, **pll), self.net.ts)

    def reset(self):
        self.pdroop_ctl.reset()
        self.qdroop_ctl.reset()
        self.pll.reset()

    def calculate(self):
        _, _, phase = self.pll.step(self.v)
        return dict(i_ref=dq0_to_abc(self.i_ref, phase))


class MasterInverter(Inverter):
    def __init__(self, v_ref=(1, 0, 0), pdroop=None, qdroop=None, **kwargs):
        self.v_ref = v_ref
        super().__init__(out_calc=dict(i_ref=3, v_ref=3), **kwargs)
        pdroop = {**(pdroop or {}), **dict(gain=40000.0, tau=.005)}
        qdroop = {**(qdroop or {}), **dict(gain=1000.0, tau=.002)}

        self.pdroop_ctl = DroopController(DroopParams(nom_value=self.net.freq_nom, **pdroop), self.net.ts)
        self.qdroop_ctl = DroopController(DroopParams(nom_value=self.net.v_nom, **qdroop), self.net.ts)
        self.dds = DDS(self.net.ts)

    def reset(self):
        self.pdroop_ctl.reset()
        self.qdroop_ctl.reset()
        self.dds.reset()

    def calculate(self):
        instPow = -inst_power(self.v, self.i)
        freq = self.pdroop_ctl.step(instPow)
        # Get the next phase rotation angle to implement
        phase = self.dds.step(freq)

        instQ = -inst_reactive(self.v, self.i)
        v_refd = self.qdroop_ctl.step(instQ)
        v_refdq0 = np.array([v_refd, 0, 0]) * self.v_ref

        return dict(i_ref=dq0_to_abc(self.i_ref, phase), v_ref=dq0_to_abc(v_refdq0, phase))

    def normalize(self, calc_data):
        super().normalize(calc_data),
        calc_data['v_ref'] /= self.v_lim


class Load(Component):
    def __init__(self, i=None, **kwargs):
        self.i = i
        super().__init__(**kwargs)

    def params(self, actions):
        # TODO: perhaps provide modelparams that set resistance value
        return super().params(actions)
