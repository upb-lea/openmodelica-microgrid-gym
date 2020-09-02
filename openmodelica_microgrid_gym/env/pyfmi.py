import logging
from datetime import datetime
from os.path import basename
import numpy as np

from pyfmi import load_fmu
from pyfmi.fmi import FMUModelME2

logger = logging.getLogger(__name__)


class PyFMI_Wrapper:
    """ convenience class"""

    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, path):
        model_name = basename(path)
        logger.debug('Loading model "%s"', model_name)
        model = cls(load_fmu(path, log_file_name=datetime.now().strftime(f'%Y-%m-%d_{model_name}.txt')))
        logger.debug('Successfully loaded model "%s"', model_name)
        return model

    def setup(self, time_start, output_names, model_params):
        self.model.reset()
        self.model.setup_experiment(start_time=time_start)

        # This is needed, because otherwise setting new values seems not to work
        #self.model.enter_initialization_mode()
        if model_params:
            values = {var: f(time_start) for var, f in model_params.items()}
            # list of keys and list of values
            self.set_params(**values)
        #self.model.exit_initialization_mode()

        e_info = self.model.get_event_info()
        e_info.newDiscreteStatesNeeded = True
        # Event iteration
        while e_info.newDiscreteStatesNeeded:
            self.model.enter_event_mode()
            self.model.event_update()
            e_info = self.model.get_event_info()

        self.model.enter_continuous_time_mode()

        # precalculating indices for more efficient lookup
        self.model_output_idx = np.array([self.model.get_variable_valueref(k) for k in output_names])

    @property
    def obs(self):
        return self.model.get_real(self.model_output_idx)

    @property
    def states(self):
        return self.model.continuous_states

    @states.setter
    def states(self, val):
        self.model.continuous_states = val

    @property
    def deriv(self):
        return self.model.get_derivatives()

    @property
    def time(self):
        return self.model.time

    @time.setter
    def time(self, val):
        self.model.time = val

    def jacc(self):
        # get state and derivative value reference lists
        refs = [[s.value_reference for s in getattr(self.model, attr)().values()]
                for attr in
                ['get_states_list', 'get_derivatives_list']]
        jacobian = np.identity(len(refs[1]))
        np.apply_along_axis(lambda col: self.model.get_directional_derivative(*refs, col), 0, jacobian)
        return jacobian

    def set(self, **kwargs):
        self.model.set(*zip(*kwargs.items()))

    def set_params(self, **kwargs):
        #self.model.enter_initialization_mode()
        self.model.initialize()                 # replacing enter and exit mode -> works to set parameters during
        # simulation AND model get outputs
        self.model.set(*zip(*kwargs.items()))
        #self.model.exit_initialization_mode()
