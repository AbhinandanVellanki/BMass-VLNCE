import abc
from typing import Any

from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import CategoricalNet

from vlnce_baselines.models.utils import CustomFixedCategorical


class ILPolicy(Policy, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions):
        """Defines an imitation learning policy as having functions act() and
        build_distribution().
        """
        super(Policy, self).__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        # Unpack potential extra_outputs from model forward (same as build_distribution)
        result = self.net(
            observations, rnn_states, prev_actions, masks
        )
        if isinstance(result, tuple) and len(result) == 3:
            features, rnn_states, extra_outputs = result
        else:
            features, rnn_states = result
            
        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        return action, rnn_states


    def get_value(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def evaluate_actions(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def build_distribution(
        self, observations, rnn_states, prev_actions, masks
    ):
        # Unpack extra_outputs from model forward
        result = self.net(observations, rnn_states, prev_actions, masks)
        if isinstance(result, tuple) and len(result) == 3:
            features, rnn_states, extra_outputs = result
        else:
            features, rnn_states = result
            extra_outputs = None
        dist = self.action_distribution(features)
        return AuxDistribution(dist, extra_outputs)

class AuxDistribution:
    def __init__(self, dist, extra_outputs):
        self._dist = dist
        self.extra_outputs = extra_outputs
        # Copy all attributes from dist
        for k, v in dist.__dict__.items():
            setattr(self, k, v)
        # Ensure logits is accessible
        if hasattr(dist, 'logits'):
            self.logits = dist.logits
    def __getattr__(self, name):
        return getattr(self._dist, name)
