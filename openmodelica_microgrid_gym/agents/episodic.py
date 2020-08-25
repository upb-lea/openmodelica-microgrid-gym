from openmodelica_microgrid_gym.agents import Agent


class EpisodicLearnerAgent(Agent):

    def observe(self, reward: float, terminated: bool):
        """
        Observes current reward.
        The idea of the episodic learner is to only learn after termination, therefore update_params() must not be
        called during the episode but only at the end.
        """
        if terminated:
            self.update_params()

    def update_params(self):
        pass

    @property
    def performance(self):
        return None
