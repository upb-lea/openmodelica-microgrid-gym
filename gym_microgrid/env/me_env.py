from .base_env import ModelicaBaseEnv, FMIStandardVersion
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ModelicaMEEnv(ModelicaBaseEnv):
    """
    Wrapper class of ModelicaBaseEnv for convenient creation of environments that utilize
    FMU exported in model exchange mode.
    Should be used as a superclass for all such environments.
    Implements abstract logic, handles different Modelica types (JModelica, Dymola) particularities.

    """

    def __init__(self, model_path, config, log_level, simulation_start_time=0):
        """

        :param model_path: path to the model FMU. Absolute path is advised.
        :param config: dictionary with model specifications. For more details see ModelicaBaseEnv docs
        :param fmi_version: version of FMI standard used in FMU compilation.
        :param log_level: level of logging to be used in experiments on environment.
        """
        self.simulation_start_time = simulation_start_time
        #    self.fmi_version = fmi_version
        logger.setLevel(log_level)
        super().__init__(model_path, "ME", config, log_level)

    def reset(self):
        """
        OpenAI Gym API. Restarts environment and sets it ready for experiments.
        In particular, does the following:
            * resets model
            * sets simulation start time to 0
            * sets initial parameters of the model
            * initializes the model
            * sets environment class attributes, e.g. start and stop time.
        :return: state of the environment after resetting
        """
        logger.debug("Experiment reset was called. Resetting the model.")

        self.model.reset()
        self.model.setup_experiment(start_time=0)

        self._set_init_parameter()

        self.start = 0
        self.stop = self.simulation_start_time

        self.state = self.do_simulation()

        self.start = self.simulation_start_time
        self.stop = self.start + self.tau
        self.done = self._is_done()
        return np.array(self.state)

    def getTime(self):
        return self.stop

