# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .layer import ResidualBlock, conv3x3, mlp
from .stu_layer import MiniSTU


# Down_sample observations before representation network (See paper appendix Network Architecture)
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.resblocks1 = nn.ModuleList(
            [ResidualBlock(out_channels // 2, out_channels // 2) for _ in range(1)]
        )
        self.conv2 = nn.Conv2d(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.downsample_block = ResidualBlock(out_channels // 2, out_channels, downsample=self.conv2, stride=2)
        self.resblocks2 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels) for _ in range(1)]
        )
        self.pooling1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.resblocks3 = nn.ModuleList(
            [ResidualBlock(out_channels, out_channels) for _ in range(1)]
        )
        self.pooling2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        for block in self.resblocks1:
            x = block(x)
        x = self.downsample_block(x)
        for block in self.resblocks2:
            x = block(x)
        x = self.pooling1(x)
        for block in self.resblocks3:
            x = block(x)
        x = self.pooling2(x)
        return x


# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):
    def __init__(self, observation_shape, num_blocks, num_channels, downsample):
        """
        Representation network
        :param observation_shape: tuple or list, shape of observations: [C, W, H]
        :param num_blocks: int, number of res blocks
        :param num_channels: int, channels of hidden states
        :param downsample: bool, True -> do downsampling for observations. (For board games, do not need)
        """
        super().__init__()
        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_shape[0],
                num_channels,
            )
        else:
            self.conv = conv3x3(
                observation_shape[0],
                num_channels,
            )
            self.bn = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x):
        if self.downsample:
            x = self.downsample_net(x)
        else:
            x = self.conv(x)
            x = self.bn(x)
            x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x


# Predict next hidden states given current states and actions
class DynamicsNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, action_space_size, is_continuous=False, action_embedding=False, action_embedding_dim=32):
        """
        Dynamics network
        :param num_blocks: int, number of res blocks
        :param num_channels: int, channels of hidden states
        :param action_space_size: int, action space size
        """
        super().__init__()
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        self.num_channels = num_channels
        self.action_space_size = action_space_size

        if action_embedding:
            self.conv1x1 = nn.Conv2d(action_space_size if is_continuous else 1, self.action_embedding_dim, 1)
            self.ln = nn.LayerNorm([action_embedding_dim, 6, 6])
            self.conv = conv3x3(num_channels + self.action_embedding_dim, num_channels)
        else:
            self.conv = conv3x3(num_channels + action_space_size if is_continuous else num_channels + 1, num_channels)

        self.bn = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

    def forward(self, state, action):
        # encode action
        if not self.is_continuous:
            action_place = torch.ones((
                state.shape[0],
                1,
                state.shape[2],
                state.shape[3],
            )).cuda().float()

            action_place = (
                    action[:, :, None, None] * action_place / self.action_space_size
            )
        else:
            action_place = action.reshape(*action.shape, 1, 1).repeat(1, 1, state.shape[-2], state.shape[-1])

        if self.action_embedding:
            action_place = self.conv1x1(action_place)
            action_place = self.ln(action_place)
            action_place = nn.functional.relu(action_place)

        x = torch.cat((state, action_place), dim=1)
        x = self.conv(x)
        x = self.bn(x)

        x += state
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        state = x

        return state


class DynamicsNetworkWithSTU(nn.Module):
    """
    DynamicsNetwork enhanced with Spectral Transform Unit (STU) and residual connections.

    This network applies spectral filtering to the dynamics model to potentially capture
    long-range dependencies in state transitions more effectively.
    """
    def __init__(self, num_blocks, num_channels, action_space_size, is_continuous=False,
                 action_embedding=False, action_embedding_dim=32,
                 dynamics_stu_seq_len=36, dynamics_stu_num_filters=8):
        """
        Dynamics network with STU
        :param num_blocks: int, number of res blocks
        :param num_channels: int, channels of hidden states
        :param action_space_size: int, action space size
        :param dynamics_stu_seq_len: int, sequence length for STU (should divide H*W)
        :param dynamics_stu_num_filters: int, number of spectral filters
        """
        super().__init__()
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.dynamics_stu_seq_len = dynamics_stu_seq_len
        self.dynamics_stu_num_filters = dynamics_stu_num_filters

        if action_embedding:
            self.conv1x1 = nn.Conv2d(action_space_size if is_continuous else 1, self.action_embedding_dim, 1)
            self.ln = nn.LayerNorm([action_embedding_dim, 6, 6])
            self.conv = conv3x3(num_channels + self.action_embedding_dim, num_channels)
        else:
            self.conv = conv3x3(num_channels + action_space_size if is_continuous else num_channels + 1, num_channels)

        self.bn = nn.BatchNorm2d(num_channels)

        # STU layer for state processing with residual connection
        # We'll reshape the state to [B, seq_len, feature_dim]
        # Assuming state is [B, num_channels, H, W], we'll treat H*W as sequence
        # The feature dimension will be num_channels
        # However, STU works best with fixed sequence length, so we reshape differently
        # We'll treat the channel dimension as part of the feature, and spatial as sequence

        # For 6x6 spatial (common after downsampling), seq_len=36, feature_dim=num_channels
        self.spatial_size = 36  # This will be H*W, typically 6*6=36 for Atari after downsampling

        # STU: processes [B, seq_len, feature_dim] -> [B, seq_len, feature_dim]
        # We set input_dim = output_dim = num_channels to maintain dimensions
        self.stu = MiniSTU(
            seq_len=dynamics_stu_seq_len,
            num_filters=dynamics_stu_num_filters,
            input_dim=num_channels,
            output_dim=num_channels,
            use_hankel_L=False,
            dtype=torch.float32,
            device=None
        )

        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )
        self.gn_out = nn.GroupNorm(8, num_channels)


    def forward(self, state, action):
        # Encode action (same as original DynamicsNetwork)
        if not self.is_continuous:
            action_place = torch.ones((
                state.shape[0],
                1,
                state.shape[2],
                state.shape[3],
            )).cuda().float()

            action_place = (
                    action[:, :, None, None] * action_place / self.action_space_size
            )
        else:
            action_place = action.reshape(*action.shape, 1, 1).repeat(1, 1, state.shape[-2], state.shape[-1])

        if self.action_embedding:
            action_place = self.conv1x1(action_place)
            action_place = self.ln(action_place)
            action_place = nn.functional.relu(action_place)

        # Concatenate state with action encoding
        x = torch.cat((state, action_place), dim=1)
        x = self.conv(x)
        x = self.bn(x)

        # Add residual connection to original state
        x += state
        x = nn.functional.relu(x)

        # Apply STU with residual connection
        # Save for residual
        residual = x

        # Reshape for STU: [B, C, H, W] -> [B, H*W, C]
        # where H*W should equal dynamics_stu_seq_len
        batch_size, num_channels, H, W = x.shape
        spatial_size = H * W

        # Only apply STU if spatial size matches seq_len
        # Otherwise skip STU processing (graceful fallback)
        if spatial_size == self.dynamics_stu_seq_len:
            # Reshape: [B, C, H, W] -> [B, H*W, C]
            x_seq = x.permute(0, 2, 3, 1).reshape(batch_size, spatial_size, num_channels)

            # Apply STU: [B, seq_len, feature_dim] -> [B, seq_len, feature_dim]
            x_seq = self.stu(x_seq)

            # Reshape back: [B, H*W, C] -> [B, C, H, W]
            x = x_seq.reshape(batch_size, H, W, num_channels).permute(0, 3, 1, 2)

            # Add residual connection
            x = x + residual
            x = nn.functional.relu(x)
        else:
            # Skip STU if dimensions don't match
            # Just use the input (residual acts as identity)
            x = residual

        # Apply residual blocks
        for block in self.resblocks:
            x = block(x)

        x = self.gn_out(x)
        state = x

        return state


class DynamicsNetworkBaseline(nn.Module):
    def __init__(self, num_blocks, num_channels, action_space_size, is_continuous=False, action_embedding=False, action_embedding_dim=32):
        """
        Dynamics network
        :param num_blocks: int, number of res blocks
        :param num_channels: int, channels of hidden states
        :param action_space_size: int, action space size
        """
        super().__init__()
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        self.num_channels = num_channels
        self.action_space_size = action_space_size

        if action_embedding:
            self.conv1x1 = nn.Conv2d(action_space_size if is_continuous else 1, self.action_embedding_dim, 1)
            self.ln = nn.LayerNorm([action_embedding_dim, 6, 6])
            self.conv = conv3x3(num_channels + self.action_embedding_dim, num_channels)
        else:
            self.conv = conv3x3(num_channels + action_space_size if is_continuous else num_channels + 1, num_channels)

        self.bn = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

        self.gn_out = nn.GroupNorm(8, num_channels)

    def forward(self, state, action):
        # encode action
        if not self.is_continuous:
            action_place = torch.ones((
                state.shape[0],
                1,
                state.shape[2],
                state.shape[3],
            )).cuda().float()

            action_place = (
                    action[:, :, None, None] * action_place / self.action_space_size
            )
        else:
            action_place = action.reshape(*action.shape, 1, 1).repeat(1, 1, state.shape[-2], state.shape[-1])

        if self.action_embedding:
            action_place = self.conv1x1(action_place)
            action_place = self.ln(action_place)
            action_place = nn.functional.relu(action_place)

        x = torch.cat((state, action_place), dim=1)
        x = self.conv(x)
        x = self.bn(x)

        x += state
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)

        
        x = self.gn_out(x)
        state = x

        return state

class DynamicsNetworkWithMamba(nn.Module):
    """
    DynamicsNetwork with Mamba selective SSM as spatial filter.
    Drop-in comparison for DynamicsNetworkWithSTU: same residual structure,
    but replaces the STU block with a Mamba block over spatial positions.
    """
    def __init__(self, num_blocks, num_channels, action_space_size, is_continuous=False,
                 action_embedding=False, action_embedding_dim=32,
                 d_state=16, d_conv=4, expand=1):
        super().__init__()
        from mamba_ssm import Mamba

        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        self.num_channels = num_channels
        self.action_space_size = action_space_size

        if action_embedding:
            self.conv1x1 = nn.Conv2d(action_space_size if is_continuous else 1, self.action_embedding_dim, 1)
            self.ln = nn.LayerNorm([action_embedding_dim, 6, 6])
            self.conv = conv3x3(num_channels + self.action_embedding_dim, num_channels)
        else:
            self.conv = conv3x3(num_channels + action_space_size if is_continuous else num_channels + 1, num_channels)

        self.bn = nn.BatchNorm2d(num_channels)

        # Mamba block: operates on [B, L=36, D=num_channels]
        self.mamba = Mamba(
            d_model=num_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )
        self.gn_out = nn.GroupNorm(8, num_channels)

    def forward(self, state, action):
        if not self.is_continuous:
            action_place = torch.ones((
                state.shape[0],
                1,
                state.shape[2],
                state.shape[3],
            )).cuda().float()

            action_place = (
                    action[:, :, None, None] * action_place / self.action_space_size
            )
        else:
            action_place = action.reshape(*action.shape, 1, 1).repeat(1, 1, state.shape[-2], state.shape[-1])

        if self.action_embedding:
            action_place = self.conv1x1(action_place)
            action_place = self.ln(action_place)
            action_place = nn.functional.relu(action_place)

        x = torch.cat((state, action_place), dim=1)
        x = self.conv(x)
        x = self.bn(x)

        x += state
        x = nn.functional.relu(x)

        # Mamba spatial filter with residual
        residual = x
        batch_size, num_channels, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(batch_size, H * W, num_channels)
        x_seq = self.mamba(x_seq)
        x = x_seq.reshape(batch_size, H, W, num_channels).permute(0, 3, 1, 2)
        x = x + residual
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)

        x = self.gn_out(x)
        state = x

        return state


class DynamicsNetworkWithAttention(nn.Module):
    """
    DynamicsNetwork with multi-head self-attention as spatial filter.
    Drop-in comparison for DynamicsNetworkWithSTU: same residual structure,
    but replaces the STU block with self-attention over spatial positions.
    """
    def __init__(self, num_blocks, num_channels, action_space_size, is_continuous=False,
                 action_embedding=False, action_embedding_dim=32,
                 num_heads=4):
        super().__init__()
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        self.num_channels = num_channels
        self.action_space_size = action_space_size

        if action_embedding:
            self.conv1x1 = nn.Conv2d(action_space_size if is_continuous else 1, self.action_embedding_dim, 1)
            self.ln = nn.LayerNorm([action_embedding_dim, 6, 6])
            self.conv = conv3x3(num_channels + self.action_embedding_dim, num_channels)
        else:
            self.conv = conv3x3(num_channels + action_space_size if is_continuous else num_channels + 1, num_channels)

        self.bn = nn.BatchNorm2d(num_channels)

        # Self-attention over spatial positions: [B, L=36, D=num_channels]
        self.attn_norm = nn.LayerNorm(num_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=num_channels,
            num_heads=num_heads,
            batch_first=True,
        )

        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

    def forward(self, state, action):
        if not self.is_continuous:
            action_place = torch.ones((
                state.shape[0],
                1,
                state.shape[2],
                state.shape[3],
            )).cuda().float()

            action_place = (
                    action[:, :, None, None] * action_place / self.action_space_size
            )
        else:
            action_place = action.reshape(*action.shape, 1, 1).repeat(1, 1, state.shape[-2], state.shape[-1])

        if self.action_embedding:
            action_place = self.conv1x1(action_place)
            action_place = self.ln(action_place)
            action_place = nn.functional.relu(action_place)

        x = torch.cat((state, action_place), dim=1)
        x = self.conv(x)
        x = self.bn(x)

        x += state
        x = nn.functional.relu(x)

        # Self-attention spatial filter with residual
        residual = x
        batch_size, num_channels, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(batch_size, H * W, num_channels)
        x_norm = self.attn_norm(x_seq)
        x_seq, _ = self.attn(x_norm, x_norm, x_norm)
        x = x_seq.reshape(batch_size, H, W, num_channels).permute(0, 3, 1, 2)
        x = x + residual
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        state = x

        return state


class DynamicsNetworkTemporalBaseline(nn.Module):
    """
    Temporal baseline: same architecture as DynamicsNetwork but receives the full
    history buffer. All N history states are stacked along the channel dimension
    so the conv layer sees [B, N*C + action_dim, 6, 6] and learns to extract
    temporal dependencies. The rest (BN, residual, resblocks) is unchanged.

    Forward signature: forward(history, action) where history is [B, N, C, H, W].
    """
    def __init__(self, num_blocks, num_channels, action_space_size, buffer_size=10,
                 is_continuous=False, action_embedding=False, action_embedding_dim=32):
        super().__init__()
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.buffer_size = buffer_size

        # Conv input: N*C channels (history stacked) + action encoding channels
        history_channels = buffer_size * num_channels
        if action_embedding:
            self.conv1x1 = nn.Conv2d(action_space_size if is_continuous else 1, self.action_embedding_dim, 1)
            self.ln = nn.LayerNorm([action_embedding_dim, 6, 6])
            self.conv = conv3x3(history_channels + self.action_embedding_dim, num_channels)
        else:
            self.conv = conv3x3(history_channels + (action_space_size if is_continuous else 1), num_channels)

        self.bn = nn.BatchNorm2d(num_channels)
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

    def forward(self, history, action):
        # history: [B, N, C, H, W]
        current_state = history[:, -1]  # [B, C, H, W]
        B, N, C, H, W = history.shape

        # Stack all history states along channel dim: [B, N*C, H, W]
        history_flat = history.reshape(B, N * C, H, W)

        if not self.is_continuous:
            action_place = torch.ones((B, 1, H, W)).cuda().float()
            action_place = action[:, :, None, None] * action_place / self.action_space_size
        else:
            action_place = action.reshape(*action.shape, 1, 1).repeat(1, 1, H, W)

        if self.action_embedding:
            action_place = self.conv1x1(action_place)
            action_place = self.ln(action_place)
            action_place = nn.functional.relu(action_place)

        x = torch.cat((history_flat, action_place), dim=1)
        x = self.conv(x)
        x = self.bn(x)

        x += current_state
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x


class DynamicsNetworkSpatioTemporalSTU(nn.Module):
    """
    Dynamics network with both spatial and temporal STU.
    - Spatial STU: filters across 36 spatial positions (proven to stabilize rollouts)
    - Temporal STU: filters across N timesteps per spatial position (captures sequential deps)
    Both applied with residual connections in sequence.

    Forward signature: forward(history, action) where history is [B, N, C, H, W].
    """
    def __init__(self, num_blocks, num_channels, action_space_size, buffer_size=10,
                 is_continuous=False, action_embedding=False, action_embedding_dim=32,
                 dynamics_stu_seq_len=36, dynamics_stu_num_filters=2):
        super().__init__()
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.buffer_size = buffer_size
        self.dynamics_stu_seq_len = dynamics_stu_seq_len

        if action_embedding:
            self.conv1x1 = nn.Conv2d(action_space_size if is_continuous else 1, self.action_embedding_dim, 1)
            self.ln = nn.LayerNorm([action_embedding_dim, 6, 6])
            self.conv = conv3x3(num_channels + self.action_embedding_dim, num_channels)
        else:
            self.conv = conv3x3(num_channels + action_space_size if is_continuous else num_channels + 1, num_channels)

        self.bn = nn.BatchNorm2d(num_channels)

        # Spatial STU: [B, 36, 64] over spatial positions
        self.spatial_stu = MiniSTU(
            seq_len=dynamics_stu_seq_len,
            num_filters=dynamics_stu_num_filters,
            input_dim=num_channels,
            output_dim=num_channels,
            use_hankel_L=False,
            dtype=torch.float32,
            device=None,
        )

        # Temporal STU: [B*36, N, 64] over timesteps per spatial position
        self.temporal_stu = MiniSTU(
            seq_len=buffer_size,
            num_filters=dynamics_stu_num_filters,
            input_dim=num_channels,
            output_dim=num_channels,
            use_hankel_L=False,
            dtype=torch.float32,
            device=None,
        )

        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

    def forward(self, history, action):
        # history: [B, N, C, H, W], action: [B, 1]
        current_state = history[:, -1]  # [B, C, H, W]

        if not self.is_continuous:
            action_place = torch.ones((
                current_state.shape[0], 1,
                current_state.shape[2], current_state.shape[3],
            )).cuda().float()
            action_place = action[:, :, None, None] * action_place / self.action_space_size
        else:
            action_place = action.reshape(*action.shape, 1, 1).repeat(
                1, 1, current_state.shape[-2], current_state.shape[-1])

        if self.action_embedding:
            action_place = self.conv1x1(action_place)
            action_place = self.ln(action_place)
            action_place = nn.functional.relu(action_place)

        x = torch.cat((current_state, action_place), dim=1)
        x = self.conv(x)
        x = self.bn(x)

        x += current_state
        x = nn.functional.relu(x)

        # 1) Spatial STU block (same as DynamicsNetworkWithSTU)
        residual = x
        B, C, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, 36, 64]
        x_seq = self.spatial_stu(x_seq)                        # [B, 36, 64]
        x = x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)    # [B, C, H, W]
        x = x + residual
        x = nn.functional.relu(x)

        # Save pre-temporal state for buffer (closer to raw states than final output)
        pre_temporal = x

        # 2) Temporal STU block (per spatial position over history)
        residual = x
        N = history.shape[1]
        h = history.permute(0, 1, 3, 4, 2)  # [B, N, H, W, C]
        h = h.permute(0, 2, 3, 1, 4)        # [B, H, W, N, C]
        h = h.reshape(B * H * W, N, C)      # [B*36, N, 64]
        h = self.temporal_stu(h)             # [B*36, N, 64]
        h = h[:, -1, :]                      # [B*36, 64]
        temporal_context = h.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        x = temporal_context + residual
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)

        # Return (output, buffer_state): output is the prediction,
        # buffer_state is what should go into the history buffer during rollout
        return x, pre_temporal


class DynamicsNetworkTemporalSTU(nn.Module):
    """
    Dynamics network with temporal STU: processes a history buffer of past states
    through MiniSTU over the time dimension (per spatial position).

    Forward signature: forward(history, action) where history is [B, N, C, H, W].
    """
    def __init__(self, num_blocks, num_channels, action_space_size, buffer_size=10,
                 is_continuous=False, action_embedding=False, action_embedding_dim=32,
                 dynamics_stu_num_filters=2):
        super().__init__()
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.buffer_size = buffer_size

        if action_embedding:
            self.conv1x1 = nn.Conv2d(action_space_size if is_continuous else 1, self.action_embedding_dim, 1)
            self.ln = nn.LayerNorm([action_embedding_dim, 6, 6])
            self.conv = conv3x3(num_channels + self.action_embedding_dim, num_channels)
        else:
            self.conv = conv3x3(num_channels + action_space_size if is_continuous else num_channels + 1, num_channels)

        self.bn = nn.BatchNorm2d(num_channels)

        # Temporal STU: processes [B*36, N, 64] per spatial position over time
        self.temporal_stu = MiniSTU(
            seq_len=buffer_size,
            num_filters=dynamics_stu_num_filters,
            input_dim=num_channels,
            output_dim=num_channels,
            use_hankel_L=False,
            dtype=torch.float32,
            device=None,
        )

        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

    def forward(self, history, action):
        # history: [B, N, C, H, W], action: [B, 1]
        current_state = history[:, -1]  # [B, C, H, W]

        # Action encoding (same as other variants)
        if not self.is_continuous:
            action_place = torch.ones((
                current_state.shape[0], 1,
                current_state.shape[2], current_state.shape[3],
            )).cuda().float()
            action_place = action[:, :, None, None] * action_place / self.action_space_size
        else:
            action_place = action.reshape(*action.shape, 1, 1).repeat(
                1, 1, current_state.shape[-2], current_state.shape[-1])

        if self.action_embedding:
            action_place = self.conv1x1(action_place)
            action_place = self.ln(action_place)
            action_place = nn.functional.relu(action_place)

        x = torch.cat((current_state, action_place), dim=1)
        x = self.conv(x)
        x = self.bn(x)

        x += current_state
        x = nn.functional.relu(x)

        # Temporal STU block
        residual = x
        B, C, H, W = x.shape
        N = history.shape[1]

        # Reshape history for per-spatial-position temporal processing
        # [B, N, C, H, W] -> [B, N, H, W, C] -> [B, H, W, N, C] -> [B*H*W, N, C]
        h = history.permute(0, 1, 3, 4, 2)  # [B, N, H, W, C]
        h = h.permute(0, 2, 3, 1, 4)        # [B, H, W, N, C]
        h = h.reshape(B * H * W, N, C)      # [B*36, N, 64]

        h = self.temporal_stu(h)             # [B*36, N, 64]
        h = h[:, -1, :]                      # [B*36, 64] — last timestep output

        temporal_context = h.reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        x = temporal_context + residual
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x


class DynamicsNetworkTemporalMamba(nn.Module):
    """
    Dynamics network with temporal Mamba: processes a history buffer of past states
    through Mamba SSM over the time dimension (per spatial position).
    """
    def __init__(self, num_blocks, num_channels, action_space_size, buffer_size=10,
                 is_continuous=False, action_embedding=False, action_embedding_dim=32,
                 d_state=16, d_conv=4, expand=1):
        super().__init__()
        from mamba_ssm import Mamba

        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.buffer_size = buffer_size

        if action_embedding:
            self.conv1x1 = nn.Conv2d(action_space_size if is_continuous else 1, self.action_embedding_dim, 1)
            self.ln = nn.LayerNorm([action_embedding_dim, 6, 6])
            self.conv = conv3x3(num_channels + self.action_embedding_dim, num_channels)
        else:
            self.conv = conv3x3(num_channels + action_space_size if is_continuous else num_channels + 1, num_channels)

        self.bn = nn.BatchNorm2d(num_channels)

        # Temporal Mamba: processes [B*36, N, 64] per spatial position over time
        self.temporal_mamba = Mamba(
            d_model=num_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

    def forward(self, history, action):
        current_state = history[:, -1]

        if not self.is_continuous:
            action_place = torch.ones((
                current_state.shape[0], 1,
                current_state.shape[2], current_state.shape[3],
            )).cuda().float()
            action_place = action[:, :, None, None] * action_place / self.action_space_size
        else:
            action_place = action.reshape(*action.shape, 1, 1).repeat(
                1, 1, current_state.shape[-2], current_state.shape[-1])

        if self.action_embedding:
            action_place = self.conv1x1(action_place)
            action_place = self.ln(action_place)
            action_place = nn.functional.relu(action_place)

        x = torch.cat((current_state, action_place), dim=1)
        x = self.conv(x)
        x = self.bn(x)

        x += current_state
        x = nn.functional.relu(x)

        # Temporal Mamba block
        residual = x
        B, C, H, W = x.shape
        N = history.shape[1]

        h = history.permute(0, 1, 3, 4, 2)  # [B, N, H, W, C]
        h = h.permute(0, 2, 3, 1, 4)        # [B, H, W, N, C]
        h = h.reshape(B * H * W, N, C)      # [B*36, N, 64]

        h = self.temporal_mamba(h)           # [B*36, N, 64]
        h = h[:, -1, :]                      # [B*36, 64]

        temporal_context = h.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = temporal_context + residual
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x


class DynamicsNetworkTemporalAttention(nn.Module):
    """
    Dynamics network with temporal attention: processes a history buffer of past states
    through multi-head self-attention over the time dimension (per spatial position).
    """
    def __init__(self, num_blocks, num_channels, action_space_size, buffer_size=10,
                 is_continuous=False, action_embedding=False, action_embedding_dim=32,
                 num_heads=4):
        super().__init__()
        self.is_continuous = is_continuous
        self.action_embedding = action_embedding
        self.action_embedding_dim = action_embedding_dim
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.buffer_size = buffer_size

        if action_embedding:
            self.conv1x1 = nn.Conv2d(action_space_size if is_continuous else 1, self.action_embedding_dim, 1)
            self.ln = nn.LayerNorm([action_embedding_dim, 6, 6])
            self.conv = conv3x3(num_channels + self.action_embedding_dim, num_channels)
        else:
            self.conv = conv3x3(num_channels + action_space_size if is_continuous else num_channels + 1, num_channels)

        self.bn = nn.BatchNorm2d(num_channels)

        # Temporal attention: processes [B*36, N, 64] per spatial position over time
        self.attn_norm = nn.LayerNorm(num_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=num_channels,
            num_heads=num_heads,
            batch_first=True,
        )

        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )

    def forward(self, history, action):
        current_state = history[:, -1]

        if not self.is_continuous:
            action_place = torch.ones((
                current_state.shape[0], 1,
                current_state.shape[2], current_state.shape[3],
            )).cuda().float()
            action_place = action[:, :, None, None] * action_place / self.action_space_size
        else:
            action_place = action.reshape(*action.shape, 1, 1).repeat(
                1, 1, current_state.shape[-2], current_state.shape[-1])

        if self.action_embedding:
            action_place = self.conv1x1(action_place)
            action_place = self.ln(action_place)
            action_place = nn.functional.relu(action_place)

        x = torch.cat((current_state, action_place), dim=1)
        x = self.conv(x)
        x = self.bn(x)

        x += current_state
        x = nn.functional.relu(x)

        # Temporal attention block
        residual = x
        B, C, H, W = x.shape
        N = history.shape[1]

        h = history.permute(0, 1, 3, 4, 2)  # [B, N, H, W, C]
        h = h.permute(0, 2, 3, 1, 4)        # [B, H, W, N, C]
        h = h.reshape(B * H * W, N, C)      # [B*36, N, 64]

        h_norm = self.attn_norm(h)
        h, _ = self.attn(h_norm, h_norm, h_norm)  # [B*36, N, 64]
        h = h[:, -1, :]                            # [B*36, 64]

        temporal_context = h.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = temporal_context + residual
        x = nn.functional.relu(x)

        for block in self.resblocks:
            x = block(x)
        return x


class DynamicsNetworkSTUSequential(nn.Module):
    """
    STU as the primary temporal dynamics model.

    Unlike previous variants that add STU as a residual on top of a CNN dynamics
    model, this uses STU as the core computation. The STU processes a temporal
    sequence of encoded states and learns the dynamics operator via spectral
    filtering over the time dimension — directly leveraging its ability to
    efficiently learn powers of the transition operator (A^n).

    Forward signature: forward(history, action) where history is [B, N, C, H, W].
    """
    def __init__(self, num_blocks, num_channels, action_space_size, buffer_size=10,
                 is_continuous=False, action_embedding=False, action_embedding_dim=32,
                 dynamics_stu_num_filters=2, hidden_dim=512, num_stu_layers=2):
        super().__init__()
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.buffer_size = buffer_size
        self.hidden_dim = hidden_dim
        self.state_dim = num_channels * 6 * 6  # 64 * 36 = 2304

        # Encode each state from flat 2304 to hidden_dim
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Action embedding
        self.action_embed = nn.Embedding(action_space_size, hidden_dim)

        # Stacked STU layers for temporal processing
        self.stu_layers = nn.ModuleList()
        self.stu_norms = nn.ModuleList()
        for _ in range(num_stu_layers):
            self.stu_layers.append(
                MiniSTU(
                    seq_len=buffer_size,
                    num_filters=dynamics_stu_num_filters,
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    use_hankel_L=False,
                    dtype=torch.float32,
                    device=None,
                )
            )
            self.stu_norms.append(nn.LayerNorm(hidden_dim))

        # Decode back to state space
        self.state_decoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),  # concat with action
            nn.ReLU(),
            nn.Linear(hidden_dim, self.state_dim),
        )

    def forward(self, history, action):
        # history: [B, N, C, H, W], action: [B, 1]
        B, N, C, H, W = history.shape
        last_state = history[:, -1]  # [B, C, H, W]

        # Flatten each state in history and encode
        states_flat = history.reshape(B, N, -1)  # [B, N, 2304]
        x = self.state_encoder(states_flat)  # [B, N, hidden_dim]

        # Apply STU layers with residual connections
        for stu, norm in zip(self.stu_layers, self.stu_norms):
            residual = x
            x = stu(x)  # [B, N, hidden_dim]
            x = norm(x + residual)

        # Take last temporal position
        x_last = x[:, -1, :]  # [B, hidden_dim]

        # Action embedding
        action_idx = action.squeeze(-1).long()  # [B]
        a_emb = self.action_embed(action_idx)  # [B, hidden_dim]

        # Decode to state space
        combined = torch.cat([x_last, a_emb], dim=-1)  # [B, hidden_dim*2]
        output = self.state_decoder(combined)  # [B, 2304]
        output = output.reshape(B, C, H, W)

        # Residual from last input state
        output = output + last_state

        return output


class STUResBlock(nn.Module):
    """Pre-norm transformer-style block with STU as the mixer.

    sublayer 1:  x = x + STU(LayerNorm(x))
    sublayer 2:  x = x + MLP(LayerNorm(x))

    The STU operates along the sequence (time) dimension. The MLP operates
    per-position. This is the JAX-repo-shaped block: STU replaces attention.
    """
    def __init__(self, d_model: int, seq_len: int, num_filters: int,
                 mlp_ratio: float = 2.0, use_hankel_L: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.stu = MiniSTU(
            seq_len=seq_len,
            num_filters=num_filters,
            input_dim=d_model,
            output_dim=d_model,
            use_hankel_L=use_hankel_L,
            dtype=torch.float32,
            device=None,
        )
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.stu(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DynamicsNetworkPureSTU(nn.Module):
    """JAX-faithful temporal STU dynamics function.

    STU is the *only* sequence mixer. There are no CNN ResBlocks, no
    flattened-spatial STU, no parallel branches. Each per-frame state is
    encoded with a single Linear, the action is appended as the (N+1)-th
    token in the sequence, a stack of STUResBlock layers mixes along time,
    and the last position is decoded back to the next-state prediction.

    Forward signature: forward(history, action) where history is [B, N, C, H, W].
    """
    def __init__(self, num_channels: int, action_space_size: int,
                 buffer_size: int = 32,
                 d_model: int = 128,
                 num_stu_layers: int = 4,
                 num_filters: int = 8,
                 mlp_ratio: float = 2.0,
                 use_hankel_L: bool = False,
                 is_continuous: bool = False,
                 # Accepted but unused — kept for build_dynamics_network compat:
                 action_embedding: bool = True,
                 action_embedding_dim: int = 16):
        super().__init__()
        self.is_continuous = is_continuous
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.buffer_size = buffer_size
        self.d_model = d_model
        self.num_filters = num_filters
        self.state_dim = num_channels * 6 * 6  # 64 * 36 = 2304

        # Per-frame encoder: project flat state to d_model. One linear, no MLP.
        self.state_encoder = nn.Linear(self.state_dim, d_model)

        # Action token: embedding (discrete) or linear projection (continuous)
        if is_continuous:
            self.action_proj = nn.Linear(action_space_size, d_model)
        else:
            self.action_embed = nn.Embedding(action_space_size, d_model)

        # Sequence has buffer_size past states + 1 action token
        self.seq_len = buffer_size + 1

        # Stack of STU+MLP residual blocks
        self.layers = nn.ModuleList([
            STUResBlock(
                d_model=d_model,
                seq_len=self.seq_len,
                num_filters=num_filters,
                mlp_ratio=mlp_ratio,
                use_hankel_L=use_hankel_L,
            )
            for _ in range(num_stu_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

        # Decoder: project last-position output back to flat state
        self.state_decoder = nn.Linear(d_model, self.state_dim)

    def forward(self, history: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # history: [B, N, C, H, W], action: [B, 1] (discrete) or [B, A] (continuous)
        B, N, C, H, W = history.shape

        # Encode each past state into a token in d_model space
        x = self.state_encoder(history.reshape(B, N, -1))  # [B, N, d_model]

        # Encode action and append as last token
        if self.is_continuous:
            a = self.action_proj(action)
        else:
            a = self.action_embed(action.squeeze(-1).long())
        x = torch.cat([x, a.unsqueeze(1)], dim=1)  # [B, N+1, d_model]

        # Stack of STU residual blocks, mixing along time
        for layer in self.layers:
            x = layer(x)

        # Read out the last position (the action token), normalize, decode
        x_last = self.norm_out(x[:, -1, :])  # [B, d_model]
        out = self.state_decoder(x_last).reshape(B, C, H, W)
        return out


class DynamicsNetworkSTUDecoder(nn.Module):
    """STU as a sequence-to-sequence dynamics decoder.

    The cleanest mapping of STU's theoretical regime onto world models:
    given an initial state s_0 and an action sequence [a_0, ..., a_{K-1}],
    predict the entire trajectory [s_1, ..., s_K] in a SINGLE forward pass
    (no autoregressive rollout, no access to intermediate ground-truth states).

    Architecture:
        encode(s_0)         -> token 0    [B, 1, d_model]
        encode(a_0..a_{K-1})-> tokens 1..K [B, K, d_model]
        concat                              [B, K+1, d_model]
        stack of STUResBlock layers (mixing along the K+1 axis)
        per-position decoder linear -> [B, K+1, state_dim]
        return positions 1..K reshaped to [B, K, C, H, W]

    The "input sequence" to STU is (initial state, action sequence). The
    "output sequence" is the predicted future trajectory. Each output state
    is causally a function of s_0 and the action prefix that produced it,
    which is exactly what an LDS impulse response represents — and what
    the Hankel eigenbasis is provably optimal for.

    Forward signature: forward(initial_state, action_sequence)
        initial_state:   [B, C, H, W]
        action_sequence: [B, K]   (discrete) or [B, K, A] (continuous)
        returns:         [B, K, C, H, W]   predicted s_1..s_K
    """
    def __init__(self, num_channels: int, action_space_size: int,
                 max_action_seq_len: int = 64,
                 d_model: int = 128,
                 num_stu_layers: int = 4,
                 num_filters: int = 8,
                 mlp_ratio: float = 2.0,
                 use_hankel_L: bool = False,
                 is_continuous: bool = False,
                 # Accepted but unused — kept for build_dynamics_network compat:
                 action_embedding: bool = True,
                 action_embedding_dim: int = 16):
        super().__init__()
        self.is_continuous = is_continuous
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.max_action_seq_len = max_action_seq_len
        self.d_model = d_model
        self.num_filters = num_filters
        self.state_dim = num_channels * 6 * 6  # 64 * 36 = 2304

        # Per-frame state encoder: project flat state to d_model
        self.state_encoder = nn.Linear(self.state_dim, d_model)

        # Action token: embedding (discrete) or linear projection (continuous)
        if is_continuous:
            self.action_proj = nn.Linear(action_space_size, d_model)
        else:
            self.action_embed = nn.Embedding(action_space_size, d_model)

        # Sequence has 1 (state) + max_action_seq_len (actions) tokens
        self.seq_len = 1 + max_action_seq_len

        # Stack of STU+MLP residual blocks
        self.layers = nn.ModuleList([
            STUResBlock(
                d_model=d_model,
                seq_len=self.seq_len,
                num_filters=num_filters,
                mlp_ratio=mlp_ratio,
                use_hankel_L=use_hankel_L,
            )
            for _ in range(num_stu_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

        # Per-position decoder: project each output position back to flat state
        self.state_decoder = nn.Linear(d_model, self.state_dim)

    def forward(self, initial_state: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        """Single-shot multistep state prediction.

        initial_state:   [B, C, H, W]
        action_sequence: [B, K] discrete int or [B, K, A] continuous
        returns:         [B, K, C, H, W]
        """
        B, C, H, W = initial_state.shape
        K = action_sequence.shape[1]

        # Encode initial state to one token
        s0_token = self.state_encoder(initial_state.reshape(B, -1))         # [B, d_model]
        s0_token = s0_token.unsqueeze(1)                                    # [B, 1, d_model]

        # Encode each action to a token
        if self.is_continuous:
            action_tokens = self.action_proj(action_sequence)               # [B, K, d_model]
        else:
            action_tokens = self.action_embed(action_sequence.long())       # [B, K, d_model]

        # Concatenate: [s0, a0, a1, ..., a_{K-1}]   shape [B, K+1, d_model]
        x = torch.cat([s0_token, action_tokens], dim=1)
        actual_seq_len = x.shape[1]

        # The STU layers were initialized with seq_len = max_action_seq_len + 1.
        # If the current sequence is shorter, pad with zeros to the model's
        # max length so the FFT-conv works at a fixed length, then truncate
        # back. This lets us use a single trained model on variable K at eval.
        if actual_seq_len < self.seq_len:
            pad_len = self.seq_len - actual_seq_len
            x = F.pad(x, (0, 0, 0, pad_len))                                # pad along seq dim

        # Stack of STU + MLP residual blocks, mixing along time
        for layer in self.layers:
            x = layer(x)

        x = self.norm_out(x)                                                # [B, seq_len, d_model]
        # Truncate back to the actual sequence length
        if actual_seq_len < self.seq_len:
            x = x[:, :actual_seq_len]

        # Decode each output position to a flat state, then take positions 1..K
        # Position 0 is the encoded s_0; positions 1..K are predictions for s_1..s_K.
        out_flat = self.state_decoder(x)                                    # [B, K+1, state_dim]
        preds = out_flat[:, 1:]                                             # [B, K, state_dim]
        return preds.reshape(B, K, C, H, W)


class DynamicsNetworkMLPSequential(nn.Module):
    """
    MLP-based temporal dynamics model — ablation for DynamicsNetworkSTUSequential.

    Same architecture as STU Sequential, but replaces spectral filtering with
    a simple MLP over the temporal dimension. This isolates whether the Hankel
    eigenvector basis specifically matters, or any temporal processing suffices.

    Forward signature: forward(history, action) where history is [B, N, C, H, W].
    """
    def __init__(self, num_blocks, num_channels, action_space_size, buffer_size=10,
                 is_continuous=False, action_embedding=False, action_embedding_dim=32,
                 hidden_dim=512, num_mlp_layers=2):
        super().__init__()
        self.num_channels = num_channels
        self.action_space_size = action_space_size
        self.buffer_size = buffer_size
        self.hidden_dim = hidden_dim
        self.state_dim = num_channels * 6 * 6

        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.action_embed = nn.Embedding(action_space_size, hidden_dim)

        # Replace STU with MLP over temporal dim: [B, N, hidden] -> [B, N, hidden]
        self.temporal_layers = nn.ModuleList()
        self.temporal_norms = nn.ModuleList()
        for _ in range(num_mlp_layers):
            self.temporal_layers.append(nn.Sequential(
                nn.Linear(buffer_size * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, buffer_size * hidden_dim),
            ))
            self.temporal_norms.append(nn.LayerNorm(hidden_dim))

        self.state_decoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.state_dim),
        )

    def forward(self, history, action):
        B, N, C, H, W = history.shape
        last_state = history[:, -1]

        states_flat = history.reshape(B, N, -1)
        x = self.state_encoder(states_flat)  # [B, N, hidden_dim]

        for mlp, norm in zip(self.temporal_layers, self.temporal_norms):
            residual = x
            x_flat = x.reshape(B, -1)  # [B, N*hidden_dim]
            x_flat = mlp(x_flat)  # [B, N*hidden_dim]
            x = x_flat.reshape(B, N, self.hidden_dim)
            x = norm(x + residual)

        x_last = x[:, -1, :]
        action_idx = action.squeeze(-1).long()
        a_emb = self.action_embed(action_idx)

        combined = torch.cat([x_last, a_emb], dim=-1)
        output = self.state_decoder(combined)
        output = output.reshape(B, C, H, W)
        output = output + last_state

        return output


class ValuePolicyNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, reduced_channels, flatten_size, fc_layers, value_output_size,
                 policy_output_size, init_zero, is_continuous=False, policy_distribution='beta', **kwargs):
        super().__init__()
        self.v_num = kwargs.get('v_num')
        self.resblocks = nn.ModuleList(
            [ResidualBlock(num_channels, num_channels) for _ in range(num_blocks)]
        )
        self.conv1x1_values = nn.ModuleList([nn.Conv2d(num_channels, reduced_channels, 1) for _ in range(self.v_num)])
        self.conv1x1_policy = nn.Conv2d(num_channels, reduced_channels, 1)
        self.bn_values = nn.ModuleList([nn.BatchNorm2d(reduced_channels) for _ in range(self.v_num)])
        self.bn_policy = nn.BatchNorm2d(reduced_channels)
        self.block_output_size_value = flatten_size
        self.block_output_size_policy = flatten_size
        self.fc_values = nn.ModuleList([mlp(self.block_output_size_value, fc_layers, value_output_size,
                            init_zero=False if is_continuous else init_zero) for _ in range(self.v_num)])
        self.fc_policy = mlp(self.block_output_size_policy, fc_layers if not is_continuous else [64],
                             policy_output_size, init_zero=init_zero)

        self.is_continuous = is_continuous
        self.init_std = 1.0
        self.min_std = 0.1

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)

        values = []
        for i in range(self.v_num):
            value = self.conv1x1_values[i](x)
            value = self.bn_values[i](value)
            value = nn.functional.relu(value)
            value = value.reshape(-1, self.block_output_size_value)
            value = self.fc_values[i](value)
            values.append(value)

        policy = self.conv1x1_policy(x)
        policy = self.bn_policy(policy)
        policy = nn.functional.relu(policy)
        policy = policy.reshape(-1, self.block_output_size_policy)
        policy = self.fc_policy(policy)

        if self.is_continuous:
            action_space_size = policy.shape[-1] // 2
            policy[:, :action_space_size] = 5 * torch.tanh(policy[:, :action_space_size] / 5)  # soft clamp mu
            policy[:, action_space_size:] = (torch.nn.functional.softplus(policy[:, action_space_size:] + self.init_std) + self.min_std)#.clip(0, 5)  # same as Dreamer-v3

        return torch.stack(values), policy

class SupportNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, reduced_channels, flatten_size, fc_layers, output_support_size, init_zero):
        super().__init__()
        self.flatten_size = flatten_size

        self.conv1x1 = nn.Conv2d(num_channels, reduced_channels, 1)
        self.bn = nn.BatchNorm2d(reduced_channels)
        self.fc = mlp(flatten_size, fc_layers, output_support_size, init_zero=init_zero)

    def forward(self, x):

        x = self.conv1x1(x)
        x = self.bn(x)
        x = nn.functional.relu(x)
        x = x.reshape(-1, self.flatten_size)
        x = self.fc(x)
        return x


class SupportLSTMNetwork(nn.Module):
    def __init__(self, num_blocks, num_channels, reduced_channels, flatten_size, fc_layers, output_support_size, lstm_hidden_size, init_zero):
        super().__init__()
        self.flatten_size = flatten_size

        self.conv1x1_reward = nn.Conv2d(num_channels, reduced_channels, 1)
        self.bn_reward = nn.BatchNorm2d(reduced_channels)
        self.lstm = nn.LSTM(input_size=flatten_size, hidden_size=lstm_hidden_size)
        self.bn_reward_sum = nn.BatchNorm1d(lstm_hidden_size)
        self.fc = mlp(lstm_hidden_size, fc_layers, output_support_size, init_zero=init_zero)

    def forward(self, x, hidden):

        x = self.conv1x1_reward(x)
        x = self.bn_reward(x)
        x = nn.functional.relu(x)
        x = x.reshape(-1, self.flatten_size).unsqueeze(0)
        x, hidden = self.lstm(x, hidden)
        x = x.squeeze(0)
        x = self.bn_reward_sum(x)
        x = nn.functional.relu(x)
        x = self.fc(x)
        return x, hidden


class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super().__init__()

        self.input_dim = input_dim
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.layer(x)


class ProjectionHeadNetwork(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x):
        return self.layer(x)
