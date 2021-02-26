from stable_baselines3 import DDPG

from experiments.hp_tune.agents.my_td3 import myTD3


class myDDPG(DDPG, myTD3):
    """
    Deep Deterministic Policy Gradient (DDPG) based on pytorch version from stable_baseline3.

    Additionally makes more training data accessible for logging e.g. in database

    Hint: for model.train() the train algorithm form myTD3 is used because python3 does NOT use depth-first search
    (in that case DDPG->TD3.train() would have been taken)
    See https://www.python-kurs.eu/python3_mehrfachvererbung.php "diamond_problem"
    """

    def __init__(self, *args, **kwargs):
        super(myDDPG, self).__init__(*args, **kwargs)
        # training variables for logging
        self.critic_loss_batch_mean = []  # mean of critic losses of the batch
        self.actor_loss_batch_mean = []  # mean of critic losses of the batch
        self.current_q_estimates_batch_mean = []  # Q(s,a)    (mean of the batch!)
        self.target_q_batch_mean = []  # yi = r + gamma*Q_target(s',µ_target(s')) (mean of the batch!)
