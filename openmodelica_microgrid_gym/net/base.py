from importlib import import_module
from itertools import chain
from typing import List, Dict, Optional, Union

import numexpr as ne
import numpy as np
import yaml
from more_itertools import collapse, flatten


class Component:
    def __init__(self, net: 'Network', id=None, in_vars=None, out_vars=None, out_calc=None):
        """


        :param net: Network to which component belongs to
        :param id: Component ID
        :param in_vars: Input variables to component
        :param out_vars: Output variables from component
        :param out_calc: (mapping from attr name to dimension of data vector) Adds values (e.g. references,...) to output
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

    def risk(self) -> float:
        return 0

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

    def fill_tmpl(self, state: np.ndarray):
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
        Will modify object variables (like current i of an inductor) it is called after all internal variables are set.
        Therefore the function has side-effects.
        The return value must be a dictionary whose keys match the keys of self.out_calc
        and whose values are of the length of out_calcs values.
        The returned values are hence additional values (like reference current i_ref).

        ::
            set(self.out_calc.keys()) == set(return)
            all([len(v) == self.out_calc[k] for k,v in return.items()])

        :return:
        """
        pass

    def normalize(self, calc_data):
        """
        Will modify object variables it is called after all internal variables are set.
        Therefore the function has side-effects, similarly to calculate().
        """
        pass

    def augment(self, state: np.ndarray, normalize=True):
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


class Network:
    """
    This class has two main functions:

    - :code:`load()`: load yaml files to instantiate an object structure of electronic components
    - :code:`augment()`: traverses all components and uses the data from the simulation and augments or modifies it.
    """

    def __init__(self, ts: float, v_nom: Union[int, str], freq_nom: float = 50):
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
        Structure of yaml-file:

        .. code-block:: text

            conf::             *net_params* *components*
            net_params::       <parameters passed to Network.__init__()>
            components::       components:
                                 *component*
                                 ...
                                 *component*
            component::        <key; has no semantic meaning, but needs to be unique>:
                                 *component_params*
            component_params:: cls: <ComponentCls>
                               in:
                                  <ComponentCls attr name>: <list of variablenames, see augment>
                               out:
                                  <ComponentCls attr name>: <list of variablenames, see augment>
                               <additional parameters passed to ComponentCls.__init__()>

        All 'in' and 'out' variable names together define the interaction with the environment,
        expected cardinality and order of the vector provided to the augment().

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

            # instantiate component class
            try:
                components_obj.append(comp_cls(net=self, **component))
            except AttributeError as e:
                raise AttributeError(f'{e!s}, please validate {configurl}')
        self.components = components_obj

        return self

    def risk(self) -> float:
        return np.max([comp.risk() for comp in self.components])

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

        :param state: raw state as recieved form the environment. must match the expected shape specified by :code:`in_vars()`
        :param normalize: boolean, specifying whether to normalize outputs
        :return: augmented and normalized state
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
    def components(self) -> List[Component]:
        return self._components

    @components.setter
    def components(self, val: List[Component]):
        self._components = val
        keys = self.out_vars(with_aug=False, flattened=True)
        for comp in self.components:
            comp.set_outidx(keys)

    def __getitem__(self, item):
        """
        get component by id

        :param item: name of the component
        :return:
        """
        for component in self.components:
            if component.id == item:
                return component
        raise ValueError(f'no such component named "{item}"')
