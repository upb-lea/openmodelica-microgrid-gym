from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class TrainRecorder(BaseCallback):

    def __init__(self, verbose=1):
        super(TrainRecorder, self).__init__(verbose)
        self.last_model_params = None  # self.model.policy.state_dict()
        self.params_change = []

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # asd = 1
        # ads = 2
        pass

    def _on_step(self) -> bool:
        asd = 1
        # R_training[self.n_calls, number_trails] = self.training_env.envs[0].env.history.df['r_load.resistor1.R'].iloc[-1]
        # R_training[self.n_calls-1, 0] = self.training_env.envs[0].env.history.df['r_load.resistor1.R'].iloc[-1]
        """
        R_training.append(self.training_env.envs[0].env.history.df['r_load.resistor1.R'].iloc[-1])
        i_phasor_training.append((self.training_env.envs[0].i_phasor+0.5)*net['inverter1'].i_lim)
        v_phasor_training.append((self.training_env.envs[0].v_phasor+0.5)*net['inverter1'].v_lim)

        if (self.training_env.envs[0].i_phasor)*net['inverter1'].i_lim > 15:
            asd = 1

        i_a.append(self.training_env.envs[0].env.history.df['lc.inductor1.i'].iloc[-1])
        i_b.append(self.training_env.envs[0].env.history.df['lc.inductor2.i'].iloc[-1])
        i_c.append(self.training_env.envs[0].env.history.df['lc.inductor3.i'].iloc[-1])

        v_a.append(self.training_env.envs[0].env.history.df['lc.capacitor1.v'].iloc[-1])
        v_b.append(self.training_env.envs[0].env.history.df['lc.capacitor2.v'].iloc[-1])
        v_c.append(self.training_env.envs[0].env.history.df['lc.capacitor3.v'].iloc[-1])
        # nach env.step()
        """
        return True

    def _on_rollout_end(self) -> None:
        # asd = 1

        model_params = self.model.policy.parameters_to_vector()

        if self.last_model_params is None:
            self.last_model_params = model_params
        else:
            self.params_change.append(np.float64(np.mean(self.last_model_params - model_params)))

        """model_params = self.model.policy.state_dict()
        if self.last_model_params is None:
            for key, value in model_params.items():
                self.params_change[key.replace(".", "_")] = []
        else:
            for key, value in model_params.items():
                #print(key)
                self.params_change[key.replace(".", "_")].append(th.mean((model_params[key]-self.last_model_params[key])).tolist())
        """

        self.last_model_params = model_params

        # self.model.actor.mu._modules # alle :)
        pass
