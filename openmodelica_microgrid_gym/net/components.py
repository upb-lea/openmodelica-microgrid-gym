from functools import partial
from typing import Optional

import numpy as np

from openmodelica_microgrid_gym.aux_ctl import DDS, DroopController, DroopParams, InverseDroopController, \
    InverseDroopParams, PLLParams, PLL
from openmodelica_microgrid_gym.aux_ctl.base import LimitLoadIntegral
from openmodelica_microgrid_gym.net.base import Component
from openmodelica_microgrid_gym.util import dq0_to_abc, inst_power, inst_reactive


class Inverter(Component):
    def __init__(self, u=None, i=None, i_noise: Optional[dict] = None, v=None, v_noise: Optional[dict] = None, i_nom=20,
                 i_lim=30,
                 v_lim=600, v_DC=1000,
                 i_ref=(0, 0, 0),
                 out_vars=None, **kwargs):
        """

        :param u:
        :param i:
        :param i_noise: structured like: must contain the key 'fun',
        the key 'clip' is optional and no clipping is applied if omited
        ::
            {
            'fun':
               {<np.random function name, e.g. "normal">: <dict of kwargs to be passed to the func>},
            'clip': <kwargs passed to clip>
            }

        :param v:
        :param v_noise: similar to i_noise
        :param i_nom:
        :param i_lim:
        :param v_lim:
        :param v_DC:
        :param i_ref:
        :param out_vars: implicit parameter not to be passed in the net.yaml, but calculated dynamically in :code:`Network`
        :param kwargs:
        """
        self.u = u
        self.v = v
        self.i = i
        # to feed static code analyser; vars will be set dynamically in the following loop
        self.i_noise = None
        self.v_noise = None
        for var in ['i', 'v']:
            # gets vars with reflection to create self.i_noise, self.v_noise
            noise_var = locals()[f'{var}_noise']  # type:dict
            if noise_var is None:
                fun = partial(np.zeros, len(out_vars[var]))
            else:
                key, value = [i[0] for i in zip(*noise_var['fun'].items())]
                clip_kwargs = noise_var.get('clip', dict(a_min=-float('inf'), a_max=float('inf')))
                fun = lambda: np.clip(getattr(np.random.default_rng(), key)(**value, size=len(out_vars[var])),
                                      **clip_kwargs)
            setattr(self, f'{var}_noise', fun)

        self.i_nom = i_nom
        self.i_lim = i_lim
        self.v_lim = v_lim
        self.v_DC = v_DC
        self.i_ref = i_ref
        super().__init__(**{'out_calc': dict(i_ref=3), 'out_vars': out_vars, **kwargs})
        self.limit_load_integrals = [
            LimitLoadIntegral(self.net.ts, self.net.freq_nom, i_nom=i_nom, i_lim=i_lim) for _ in
            range(3)]

    def reset(self):
        [integ.reset() for integ in self.limit_load_integrals]

    def normalize(self, calc_data):
        self.i = self.i / self.i_lim
        self.v = self.v / self.v_lim
        calc_data['i_ref'] = calc_data['i_ref'] / self.i_lim

    def risk(self) -> float:
        return max([integ.risk() for integ in self.limit_load_integrals])

    def params(self, actions):
        return {**super().params(actions), **{self._prefix_var(['.v_DC']): self.v_DC}}

    def calculate(self):
        self.i = self.i + self.i_noise()
        self.v = self.v + self.v_noise()
        [integ.step(i) for i, integ in zip(self.i, self.limit_load_integrals)]


class SlaveInverter(Inverter):
    def __init__(self, pll=None, pdroop=None, qdroop=None, **kwargs):
        super().__init__(**kwargs)

        pdroop = {**dict(gain=40000.0), **(pdroop or {})}
        qdroop = {**dict(gain=50.0), **(qdroop or {})}
        pll = {**dict(kP=10, kI=200), **(pll or {})}

        # toDo: set time Constant for droop Filter correct
        self.pdroop_ctl = InverseDroopController(
            InverseDroopParams(tau=self.net.ts, nom_value=self.net.freq_nom, **pdroop), self.net.ts)
        self.qdroop_ctl = InverseDroopController(
            InverseDroopParams(tau=self.net.ts, nom_value=self.net.v_nom, **qdroop), self.net.ts)
        # default pll params and new ones
        self.pll = PLL(PLLParams(f_nom=self.net.freq_nom, **pll), self.net.ts)

    def reset(self):
        super().reset()
        self.pdroop_ctl.reset()
        self.qdroop_ctl.reset()
        self.pll.reset()

    def calculate(self):
        super().calculate()
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
        super().reset()
        self.pdroop_ctl.reset()
        self.qdroop_ctl.reset()
        self.dds.reset()

    def calculate(self):
        super().calculate()
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
        super().reset()
        self.dds.reset()

    def calculate(self):
        super().calculate()
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
