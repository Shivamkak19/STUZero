# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

"""
Spectral EfficientZero Agent for Atari with STU layers.

This agent integrates spectral filtering to test the hypothesis that
spectral state space models can outperform standard EfficientZero on
Atari benchmarks by better handling long-range temporal dependencies.
"""

import random
import time
import copy
import math
from ez.agents.base import Agent
from omegaconf import open_dict

from ez.envs import make_atari
from ez.utils.format import DiscreteSupport
from ez.agents.models.spectral_model import SpectralEfficientZero
from ez.agents.models.base_model import *
from ez.agents.models.spectral_layers import create_spectral_value_policy_network


class SpectralEZAtariAgent(Agent):
    """
    Spectral-enhanced EfficientZero agent for Atari environments.

    Integrates Spectral Transform Units (STU) into the value-policy network
    to capture long-range temporal dependencies through spectral filtering.

    Key hypothesis: Spectral filtering provides:
    1. Robust handling of long-range dependencies independent of spectral gap
    2. Fixed convolutional filters (parameter-free)
    3. Theoretically-founded temporal modeling
    """

    def __init__(self, config):
        super().__init__(config)

        self.update_config()

        self.num_blocks = config.model.num_blocks
        self.num_channels = config.model.num_channels
        self.reduced_channels = config.model.reduced_channels
        self.fc_layers = config.model.fc_layers
        self.down_sample = config.model.down_sample
        self.state_norm = config.model.state_norm
        self.value_prefix = config.model.value_prefix
        self.init_zero = config.model.init_zero
        self.action_embedding = config.model.action_embedding
        self.action_embedding_dim = config.model.action_embedding_dim
        self.value_policy_detach = config.train.value_policy_detach
        self.use_spectral = config.model.get('use_spectral', True)
        self.num_spectral_filters = config.model.get('num_spectral_filters', 16)

    def update_config(self):
        assert not self._update

        env = make_atari(self.config.env.game, seed=0, save_path=None, **self.config.env)
        action_space_size = env.action_space.n

        obs_channel = 1 if self.config.env.gray_scale else 3

        reward_support = DiscreteSupport(self.config)
        reward_size = reward_support.size

        value_support = DiscreteSupport(self.config)
        value_size = value_support.size

        localtime = time.strftime('%Y-%m-%d %H:%M:%S')
        tag = '{}-spectral-seed={}-{}/'.format(self.config.tag, self.config.env.base_seed, localtime)

        with open_dict(self.config):
            self.config.env.action_space_size = action_space_size
            self.config.mcts.num_top_actions = min(action_space_size, self.config.mcts.num_top_actions)
            self.config.env.obs_shape[0] = obs_channel
            self.config.rl.discount **= self.config.env.n_skip
            self.config.model.reward_support.size = reward_size
            self.config.model.value_support.size = value_size

            if action_space_size < 4:
                self.config.mcts.num_top_actions = 2
                self.config.mcts.num_simulations = 4
            elif action_space_size < 16:
                self.config.mcts.num_top_actions = 4
            else:
                self.config.mcts.num_top_actions = 8

            if not self.config.mcts.use_gumbel:
                self.config.mcts.num_simulations = 50

            # Add spectral filtering config
            if not hasattr(self.config.model, 'use_spectral'):
                self.config.model.use_spectral = True
            if not hasattr(self.config.model, 'num_spectral_filters'):
                self.config.model.num_spectral_filters = 16

            print(f'[Spectral Agent] env={self.config.env.env}, game={self.config.env.game}, '
                  f'|A|={action_space_size}, top_m={self.config.mcts.num_top_actions}, '
                  f'N={self.config.mcts.num_simulations}, use_spectral={self.config.model.use_spectral}, '
                  f'K={self.config.model.num_spectral_filters}')
            self.config.save_path += tag

        self.obs_shape = copy.deepcopy(self.config.env.obs_shape)
        self.input_shape = copy.deepcopy(self.config.env.obs_shape)
        self.input_shape[0] *= self.config.env.n_stack
        self.action_space_size = self.config.env.action_space_size

        self._update = True

    def build_model(self):
        """
        Build the Spectral EfficientZero model with STU-enhanced value-policy network.
        """
        if self.down_sample:
            state_shape = (self.num_channels,
                          math.ceil(self.obs_shape[1] / 16),
                          math.ceil(self.obs_shape[2] / 16))
        else:
            state_shape = (self.num_channels, self.obs_shape[1], self.obs_shape[2])

        state_dim = state_shape[0] * state_shape[1] * state_shape[2]
        flatten_size = self.reduced_channels * state_shape[1] * state_shape[2]

        # Standard representation network
        representation_model = RepresentationNetwork(
            self.input_shape, self.num_blocks, self.num_channels, self.down_sample
        )

        # Standard dynamics network
        dynamics_model = DynamicsNetwork(
            self.num_blocks, self.num_channels, self.action_space_size,
            action_embedding=self.action_embedding,
            action_embedding_dim=self.action_embedding_dim
        )

        # Spectral-enhanced value-policy network
        value_policy_model = create_spectral_value_policy_network(
            self.config,
            use_spectral=self.use_spectral
        )

        # Standard reward prediction
        reward_output_size = self.config.model.reward_support.size
        if self.value_prefix:
            reward_prediction_model = SupportLSTMNetwork(
                0, self.num_channels, self.reduced_channels,
                flatten_size, self.fc_layers, reward_output_size,
                self.config.model.lstm_hidden_size, self.init_zero
            )
        else:
            reward_prediction_model = SupportNetwork(
                self.num_blocks, self.num_channels, self.reduced_channels,
                flatten_size, self.fc_layers, reward_output_size,
                self.init_zero
            )

        # Projection networks for consistency loss
        projection_layers = self.config.model.projection_layers
        head_layers = self.config.model.prjection_head_layers
        assert projection_layers[1] == head_layers[1]

        projection_model = ProjectionNetwork(state_dim, projection_layers[0], projection_layers[1])
        projection_head_model = ProjectionHeadNetwork(projection_layers[1], head_layers[0], head_layers[1])

        # Build Spectral EfficientZero model
        spectral_ez_model = SpectralEfficientZero(
            representation_model, dynamics_model, reward_prediction_model, value_policy_model,
            projection_model, projection_head_model, self.config,
            state_norm=self.state_norm, value_prefix=self.value_prefix,
            use_spectral=self.use_spectral
        )

        return spectral_ez_model
