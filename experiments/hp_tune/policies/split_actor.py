from typing import Optional, Tuple, List, Type

import gym
import torch as th
from torch import nn

from stable_baselines3.common.policies import register_policy, BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.td3.policies import TD3Policy, Actor


def mlp(sizes, activation, output_activation=None):
    """
    Defines a multi layer perceptron using pytorch layers and activation funtions
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        if act is not None:
            layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        else:
            layers += [nn.Linear(sizes[j], sizes[j + 1])]
    # layers.append(nn.Tanh())
    return layers


class CustomActor(Actor):
    """
    Actor network (policy) for TD3.
    """

    def __init__(self, *args, **kwargs):
        super(CustomActor, self).__init__(*args, **kwargs)
        # Define custom network with Dropout
        # WARNING: it must end with a tanh activation to squash the output
        # self.mu = nn.Sequential(*mlp([20, 10, 5, 6], nn.LeakyReLU()))

        # self.mu = nn.Sequential(nn.Linear(kwargs['observation_space'].shape[0], 32),
        #                       kwargs['activation_fn'](negative_slope=0.02),
        #                       nn.Linear(32, 10),
        #                       nn.LeakyReLU(negative_slope=0.02),
        #                       nn.Linear(10, int(kwargs['action_space'].shape[0] / 2)))
        # self.I = nn.Sequential(nn.Linear(kwargs['observation_space'].shape[0], 32),
        #                       kwargs['activation_fn'](negative_slope=0.02),
        #                       nn.Linear(32, 10),
        #                       nn.LeakyReLU(negative_slope=0.02),
        #                       nn.Linear(10, int(kwargs['action_space'].shape[0] / 2)))

        self.mu = nn.Sequential(*mlp([kwargs['observation_space'].shape[0], *kwargs['net_arch'],
                                      int(kwargs['action_space'].shape[0] / 2)],
                                     kwargs['activation_fn'],
                                     nn.Tanh))

        self.I = nn.Sequential(*mlp([kwargs['observation_space'].shape[0], *kwargs['net_arch'],
                                     int(kwargs['action_space'].shape[0] / 2)],
                                    kwargs['activation_fn'],
                                    nn.Tanh))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        # assert deterministic, 'The TD3 actor only outputs deterministic actions'
        features = self.extract_features(obs)
        return th.cat((self.mu(features), self.I(features)), 1)


class CustomContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    """

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            net_arch: List[int],
            features_extractor: nn.Module,
            features_dim: int,
            activation_fn: Type[nn.Module] = nn.ReLU,
            normalize_images: bool = True,
            n_critics: int = 2,
            share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            # q_net = create_mlp(features_dim + action_dim, 1, net_arch, activation_fn)
            # Define critic with Dropout here
            # q_net = nn.Sequential(  nn.Linear(features_dim + action_dim, 32),
            #                        nn.ReLU(),
            #                        nn.Linear(32, 10),
            #                        nn.ReLU(),
            #                        nn.Linear(10, 1)
            #                    )

            q_net = nn.Sequential(*mlp([features_dim + action_dim, *net_arch, 1],
                                       activation_fn
                                       ))

            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor using the policy loss only
        # when the features_extractor is shared with the actor
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs)
        return self.q_networks[0](th.cat([features, actions], dim=1))


class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super(CustomTD3Policy, self).__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCritic(**critic_kwargs).to(self.device)


register_policy("CustomTD3Policy", CustomTD3Policy)
