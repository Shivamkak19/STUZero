# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import numpy as np
import torch.nn as nn
from ez.utils.format import normalize_state
from ez.utils.format import formalize_obs_lst, DiscreteSupport, allocate_gpu, prepare_obs_lst, symexp


class EfficientZero(nn.Module):
    def __init__(self,
                 representation_model,
                 dynamics_model,
                 reward_prediction_model,
                 value_policy_model,
                 projection_model,
                 projection_head_model,
                 config,
                 **kwargs,
                 ):
        """The basic models in EfficientZero
        Parameters
        ----------
        representation_model: nn.Module
            represent the observations'
        dynamics_model: nn.Module
            dynamics model predicts the next state given the current state and action
        reward_prediction_model: nn.Module
            predict the reward given the next state (Namely, current state and action)
        value_prediction_model: nn.Module
            predict the value given the state
        policy_prediction_model: nn.Module
            predict the policy given the state
        kwargs: dict
            state_norm: bool.
                use state normalization for encoded state
            value_prefix: bool
                predict value prefix instead of reward
        """
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
        self.seq_len = config.model.seq_len

    def do_representation(self, obs):
        state = self.representation_model(obs)
        if self.state_norm:
            state = normalize_state(state)

        return state

    def do_dynamics(self, state_history, action_history):
        assert state_history.shape[0] == action_history.shape[0], f"Batch size of state history {state_history.shape[0]} does not match that of action history {action_history.shape[0]}"
        assert state_history.shape[1] == self.seq_len, f"Sequence length of state history {state_history.shape[1]} does not match the expected {self.seq_len}"
        assert action_history.shape[1] == self.seq_len, f"Sequence length of action history {action_history.shape[1]} does not match the expected {self.seq_len}"

        next_state = self.dynamics_model(state_history, action_history)
        if self.state_norm:
            next_state = normalize_state(next_state)

        return next_state

    def do_reward_prediction(self, next_state, reward_hidden=None):
        # use the predicted state (Namely, current state + action) for reward prediction
        if self.value_prefix:
            value_prefix, reward_hidden = self.reward_prediction_model(next_state, reward_hidden)
            return value_prefix, reward_hidden
        else:
            reward = self.reward_prediction_model(next_state)
            return reward, None

    def do_value_policy_prediction(self, state):
        value, policy = self.value_policy_model(state)
        return value, policy

    def do_projection(self, state, with_grad=True):
        # only the branch of proj + pred can share the gradients
        proj = self.projection_model(state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head_model(proj)
            return proj
        else:
            return proj.detach()

    def initial_inference(self, obs, training=False):
        state = self.do_representation(obs)
        state_history = torch.zeros(state.shape[0], self.seq_len - 1, *state.shape[1:], device=state.device, dtype=state.dtype)
        state_history = torch.cat([state_history, state.unsqueeze(1)], dim=1)
        values, policy = self.do_value_policy_prediction(state)

        if training:
            return state, state_history, values, policy

        if self.v_num > 2:
            values = values[np.random.choice(self.v_num, 2, replace=False)]
        if self.config.model.value_support.type == 'symlog':
            output_values = symexp(values).min(0)[0]
        else:
            output_values = DiscreteSupport.vector_to_scalar(values, **self.config.model.value_support).min(0)[0]

        if self.config.env.env in ['DMC', 'Gym']:
            output_values = output_values.clip(0, 1e5)

        return state, state_history, output_values, policy


    def recurrent_inference(self, history, reward_hidden, training=False):
        # history: tuple of (state_history, action_history) where we have s[i]->a[i]->s[i+1]
        state_history, action_history = history
        next_state = self.do_dynamics(state_history, action_history)
        new_state_history = torch.cat([state_history[:, 1:, :, :, :], next_state.unsqueeze(1)], dim=1)
        value_prefix, reward_hidden = self.do_reward_prediction(next_state, reward_hidden)
        values, policy = self.do_value_policy_prediction(next_state)
        if training:
            return next_state, new_state_history, value_prefix, values, policy, reward_hidden

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
        return next_state, new_state_history, value_prefix, output_values, policy, reward_hidden

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