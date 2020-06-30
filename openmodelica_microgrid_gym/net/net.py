import sys
from typing import List, Optional

import yaml

from openmodelica_microgrid_gym import ModelicaEnv
from openmodelica_microgrid_gym.aux_ctl import PLL, PLLParams


class Network:
    def __init__(self, ts, v_nom, freq_nom, i_nom, i_lim, v_DC=1000):
        self.ts = ts
        self.v_nom = v_nom
        self.freq_nom = freq_nom
        self.i_nom = i_nom
        self.i_lim = i_lim
        self.v_DC = v_DC
        self.components = []  # type: List[Component]

    @classmethod
    def load(cls, configurl):
        data = yaml.safe_load(open(configurl))
        components = data['components']
        del data['components']
        self = cls(**data)

        for name, component in components.items():
            comp_cls = component['cls']
            del component['cls']
            comp_cls = getattr(sys.modules[__name__], comp_cls)

            # rename keys (because 'in' is a reserved keyword)
            if 'in' in component:
                component['in_vars'] = component.pop('in')
            if 'out' in component:
                component['out_vars'] = component.pop('out')

            self.components.append(comp_cls(net=self, **component))

        return self

    def statekeys(self, keys):
        for comp in self.components:
            comp.set_outidx(keys)

    def getstatekeys(self):
        return [comp.outputnames() for comp in self.components]

    def augment(self, state):
        return np.hstack([comp.normalized_outputs(state) for comp in self.components])


class Component:
    def __init__(self, net: Network, id, in_vars=None, out_vars=None):
        self.net = net
        self.id = id
        self.in_vars = in_vars
        self.in_idx = None  # type: Optional[dict]

        self.out_vars = out_vars
        self.out_idx = None  # type: Optional[dict]

    def inputs(self):
        pass

    def outputs(self, state):
        self.fill_tmpl(state)
        return np.hstack([getattr(self, attr) for attr in self.out_idx.keys()])

    def outputnames(self):
        return [['.'.join([self.id, attr, i]) for i, _ in enumerate(vals)] for attr, vals in self.out_vars.items()]

    def fill_tmpl(self, state):
        if self.out_idx is None:
            raise ValueError('call set_tmplidx before fill_tmpl. the keys must be converted to indices for efficiency')
        for attr, idxs in self.out_idx.items():
            # set object variables to the respective state variables
            # TODO: maybe validate and only allow the values that the component supports: hasattr(self,attr)
            setattr(self, attr, state[idxs])

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
    def __init__(self, u=None, i=None, v=None, **kwds):
        super().__init__(**kwds)
        self.u = u
        self.v = v
        self.i = i

    # has Vmax, Imax
    pass

    def outputs(self, state):
        # calculate ABC conversion and add references to outputs
        pass


class SlaveInverter(Inverter):
    def __init__(self, iref=(0, 0, 0), **kwds):
        super().__init__(**kwds)
        self.pll = PLL(PLLParams(.02, 1, limits=(-1, 1)), self.net.ts)

    # contains a pll
    pass


class MasterInverter(Inverter):
    def __init__(self, iref=(0, 0, 0), vref=(1, 0, 0), **kwds):
        super().__init__(**kwds)


class Load(Component):
    def __init__(self, i=None, **kwds):
        super().__init__(**kwds)
        self.i = i

    def inputs(self):
        # perhaps provide modelparams that set resistance value
        return super().inputs()


if __name__ == '__main__':
    import numpy as np

    # env = ModelicaEnv()

    net = Network.load('net.yaml')
    # net.statekeys(env.model_output_names)
    net.statekeys([f'lcl1.capacitor{k}.v' for k in '123'] + [f'lc1.capacitor{k}.v' for k in '123']
                  + [f'lc1.inductor{k}.i' for k in '123'] + [f'lcl1.inductor{k}.i' for k in '123']
                  + [f'rl1.inductor{k}.i' for k in '123'])
    net.augment(np.array([1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 4, 4, 4]))
