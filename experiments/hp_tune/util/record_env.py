from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import time
from experiments.hp_tune.env.vctrl_single_inv import folder_name


class RecordEnvCallback(BaseCallback):

    def __init__(self, env, agent, max_episode_steps, recorder=None, n_trail=0):
        """
        Class to interact with stable_baseline learner callback,
        Runs e.g. every 1000 steps to evaluate the learning process in the env:

        plot_callback = EveryNTimesteps(n_steps=1000, callback=RecordEnvCallback(env, model))
        agent.learn(total_timesteps=2000, callback=[plot_callback])

        :param env: environment to run on
        :param agent: agent to run on env to evaluate
        """
        self.env = env
        self.agent = agent
        self.max_episode_steps = max_episode_steps
        self.recorder = recorder
        self.n_trail = n_trail
        super().__init__()

    def _on_step(self) -> bool:
        rewards = []
        obs = self.env.reset()
        for _ in range(self.max_episode_steps):
            self.env.render()
            action, _states = self.agent.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
            if done or self.info.get("timelimit_reached", False):
                break
        # plot rewards?

        ts = time.gmtime()

        plt.plot(rewards)
        plt.xlabel(r'$t\,/\,\mathrm{s}$')
        plt.ylabel('$Reward$')
        plt.grid(which='both')

        plt.savefig(f'{folder_name}/{self.n_trail}/Reward{time.strftime("%Y_%m_%d__%H_%M_%S", ts)}.pdf')
        plt.close()

        validation_during_training = {"Name": "Validation during training",
                                      "num_timesteps_learned_so_far": self.num_timesteps,
                                      "time": ts,
                                      "Reward": rewards}

        # Add v-measurements
        validation_during_training.update(
            {self.env.env.viz_col_tmpls[j].vars[i].replace(".", "_"): self.env.env.history[
                self.env.env.viz_col_tmpls[j].vars[i]].copy().tolist() for j in range(2) for i in range(6)
             })

        validation_during_training.update(
            {self.env.env.viz_col_tmpls[2].vars[i].replace(".", "_"): self.env.env.history[
                self.env.env.viz_col_tmpls[2].vars[i]].copy().tolist() for i in range(3)
             })

        # va = self.env.env.history[self.env.env.viz_col_tmpls[0].vars[0]].copy()

        self.recorder.save_to_mongodb('Trail_number_' + self.n_trail, validation_during_training)

        self.env.close()
        self.env.reset()
        return True
