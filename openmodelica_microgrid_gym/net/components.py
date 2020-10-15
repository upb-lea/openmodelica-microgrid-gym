from itertools import chain
from typing import Optional, Dict

import numpy as np

from openmodelica_microgrid_gym.aux_ctl import DDS, DroopController, DroopParams, InverseDroopController, \
    InverseDroopParams, PLLParams, PLL
from openmodelica_microgrid_gym.net.net import Network
from openmodelica_microgrid_gym.util import dq0_to_abc, inst_power, inst_reactive


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
            return r + [[self._prefix_var([self.id, attr, str(i)]) for i in range(n)] for attr, n in
                        self.out_calc.items()]

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
        # This pre-calculation is done mainly for performance reasons
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
        if strs[0].startswith('.'):
            # first string minus its prefix '.' and the remaining strs
            strs[0] = strs[0][1:]
            if self.id is not None:
                # this is a complete identifier like 'lc1.inductor1.i' that should not be modified:
                strs = [self.id] + strs
        return '.'.join(strs)

    def calculate(self) -> Dict[str, np.ndarray]:
        """
        will write internal variables it is called after all internal variables are set
        The return value must be a dictionary whose keys match the keys of self.out_calc and whose values are of the length of outcalcs values

        set(self.out_calc.keys()) == set(return)
        all([len(v) == self.out_calc[k] for k,v in return.items()])
        :return:
        """
        pass

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


class MasterInverterCurrentSourcing(Inverter):
    def __init__(self, f_nom=50, **kwargs):
        super().__init__(out_calc=dict(i_ref=3), **kwargs)
        self.dds = DDS(self.net.ts)
        self.f_nom = f_nom

    def reset(self):
        self.dds.reset()

    def calculate(self):
        # Get the next phase rotation angle to implement
        phase = self.dds.step(self.f_nom)
        return dict(i_ref=dq0_to_abc(self.i_ref, phase))


class Load(Component):
    def __init__(self, i=None, **kwargs):
        self.i = i
        super().__init__(**kwargs)

    def params(self, actions):
        # TODO: perhaps provide modelparams that set resistance value
        return super().params(actions)
