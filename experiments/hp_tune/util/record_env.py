from stable_baselines3.common.callbacks import BaseCallback


class RecordEnvCallback(BaseCallback):

    def __init__(self, env, agent, max_episode_steps):
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
        super().__init__()

    def _on_step(self) -> bool:
        rewards = []
        obs = self.env.reset()
        for _ in range(self.max_episode_steps):
            self.env.render()
            action, _states = self.agent.predict(obs, deterministic=True)
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
            if done:
                break
        # plot rewards?
        self.env.close()
        self.env.reset()
        return True
