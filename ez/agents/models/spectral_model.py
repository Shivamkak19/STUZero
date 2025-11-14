# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

"""
Spectral-enhanced EfficientZero model with STU layers.
"""

import torch
import torch.nn as nn
import numpy as np
from ez.utils.format import normalize_state, DiscreteSupport, symexp


class SpectralEfficientZero(nn.Module):
    """
    EfficientZero with Spectral Filtering for enhanced temporal modeling.

    This model integrates Spectral Transform Units (STU) to capture long-range
    temporal dependencies in the hidden state sequence during unrolling.
    """
    def __init__(
        self,
        representation_model,
        dynamics_model,
        reward_prediction_model,
        value_policy_model,
        projection_model,
        projection_head_model,
        config,
        **kwargs,
    ):
        super().__init__()

        self.representation_model = representation_model
        self.dynamics_model = dynamics_model
        self.reward_prediction_model = reward_prediction_model
        self.value_policy_model = value_policy_model
        self.projection_model = projection_model
        self.projection_head_model = projection_head_model
        self.config = config
        self.state_norm = kwargs.get('state_norm')
        self.value_prefix = kwargs.get('value_prefix')
        self.v_num = config.train.v_num
        self.use_spectral = kwargs.get('use_spectral', True)

        # Buffer to store state sequence during unrolling (for spectral filtering)
        self.state_sequence = None

    def do_representation(self, obs):
        state = self.representation_model(obs)
        if self.state_norm:
            state = normalize_state(state)
        return state

    def do_dynamics(self, state, action):
        next_state = self.dynamics_model(state, action)
        if self.state_norm:
            next_state = normalize_state(next_state)
        return next_state

    def do_reward_prediction(self, next_state, reward_hidden=None):
        if self.value_prefix:
            value_prefix, reward_hidden = self.reward_prediction_model(next_state, reward_hidden)
            return value_prefix, reward_hidden
        else:
            reward = self.reward_prediction_model(next_state)
            return reward, None

    def do_value_policy_prediction(self, state, state_sequence=None):
        """
        Enhanced value-policy prediction with optional spectral filtering.

        Args:
            state: Current state [B, C, H, W]
            state_sequence: Optional sequence of states for spectral filtering
                          [B, seq_len, C, H, W]
        """
        if self.use_spectral and hasattr(self.value_policy_model, 'forward'):
            # Check if the value_policy_model supports state_sequence
            import inspect
            sig = inspect.signature(self.value_policy_model.forward)
            if 'state_sequence' in sig.parameters:
                value, policy = self.value_policy_model(state, state_sequence=state_sequence)
            else:
                value, policy = self.value_policy_model(state)
        else:
            value, policy = self.value_policy_model(state)
        return value, policy

    def do_projection(self, state, with_grad=True):
        proj = self.projection_model(state)
        if with_grad:
            proj = self.projection_head_model(proj)
            return proj
        else:
            return proj.detach()

    def initial_inference(self, obs, training=False):
        """
        Initial inference step. Initializes the state sequence buffer.
        """
        state = self.do_representation(obs)

        # Initialize state sequence buffer for spectral filtering
        if self.use_spectral:
            B, C, H, W = state.shape
            self.state_sequence = state.unsqueeze(1)  # [B, 1, C, H, W]

        values, policy = self.do_value_policy_prediction(state, self.state_sequence if self.use_spectral else None)

        if training:
            return state, values, policy

        if self.v_num > 2:
            values = values[np.random.choice(self.v_num, 2, replace=False)]
        if self.config.model.value_support.type == 'symlog':
            output_values = symexp(values).min(0)[0]
        else:
            output_values = DiscreteSupport.vector_to_scalar(values, **self.config.model.value_support).min(0)[0]

        if self.config.env.env in ['DMC', 'Gym']:
            output_values = output_values.clip(0, 1e5)

        return state, output_values, policy

    def recurrent_inference(self, state, action, reward_hidden, training=False):
        """
        Recurrent inference with spectral filtering support.

        Maintains a sliding window of states for spectral filtering.
        """
        next_state = self.do_dynamics(state, action)
        value_prefix, reward_hidden = self.do_reward_prediction(next_state, reward_hidden)

        # Update state sequence for spectral filtering
        if self.use_spectral and self.state_sequence is not None:
            B, L, C, H, W = self.state_sequence.shape
            # Append new state
            next_state_expanded = next_state.unsqueeze(1)  # [B, 1, C, H, W]
            self.state_sequence = torch.cat([self.state_sequence, next_state_expanded], dim=1)

            # Keep only the most recent seq_len states
            max_seq_len = self.config.rl.unroll_steps + 1
            if self.state_sequence.shape[1] > max_seq_len:
                self.state_sequence = self.state_sequence[:, -max_seq_len:, :, :, :]

        values, policy = self.do_value_policy_prediction(next_state, self.state_sequence if self.use_spectral else None)

        if training:
            return next_state, value_prefix, values, policy, reward_hidden

        if self.v_num > 2:
            values = values[np.random.choice(self.v_num, 2, replace=False)]
        if self.config.model.value_support.type == 'symlog':
            output_values = symexp(values).min(0)[0]
        else:
            output_values = DiscreteSupport.vector_to_scalar(values, **self.config.model.value_support).min(0)[0]

        if self.config.env.env in ['DMC', 'Gym']:
            output_values = output_values.clip(0, 1e5)

        if self.config.model.reward_support.type == 'symlog':
            value_prefix = symexp(value_prefix)
        else:
            value_prefix = DiscreteSupport.vector_to_scalar(value_prefix, **self.config.model.reward_support)

        return next_state, value_prefix, output_values, policy, reward_hidden

    def reset_state_sequence(self):
        """Reset the state sequence buffer (e.g., at episode boundaries)."""
        self.state_sequence = None

    def get_weights(self, part='none'):
        if part == 'reward':
            weights = self.reward_prediction_model.state_dict()
        else:
            weights = self.state_dict()
        return {k: v.cpu() for k, v in weights.items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)
