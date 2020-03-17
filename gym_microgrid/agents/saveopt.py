from gym_microgrid.agents import Agent


class SafeOptAgent(Agent):
    def __init__(self):
        super().__init__()
        self.episode_reward = 0

    def observe(self, reward, terminated):
        self.episode_reward += reward
        if terminated:
            # safeopt update step
            # TODO
            # reset episode reward
            self.episode_reward = 0
        # on other steps we don't need to do anything
