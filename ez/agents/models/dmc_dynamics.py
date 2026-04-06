"""
DMC-specific dynamics model variants for STUZero.

All models operate on flat [B, 128] latent states (not spatial) with continuous
actions [B, action_shape]. Four variants are provided:

1. DMCDynamicsBaseline       -- exact copy of the MLP-based DynamicsNetwork
2. DMCDynamicsBaselineNorm   -- baseline + output LayerNorm (fair comparison)
3. DMCDynamicsWithSTU        -- "Feature STU": STU as a feature mixer on flat vectors
4. DMCDynamicsSTUSequential  -- temporal STU over a history buffer of past states
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ez.agents.models.stu_layer import MiniSTU

__all__ = [
    "ImproveResidualBlock",
    "DMCDynamicsBaseline",
    "DMCDynamicsBaselineNorm",
    "DMCDynamicsWithSTU",
    "DMCDynamicsSTUSequential",
]


class ImproveResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm -> Linear -> ReLU -> Linear + skip."""

    def __init__(self, input_shape: int, hidden_shape: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(input_shape)
        self.linear1 = nn.Linear(input_shape, hidden_shape)
        self.linear2 = nn.Linear(hidden_shape, input_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.ln1(x)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        out += identity
        return out


# ---------------------------------------------------------------------------
# 1. Baseline -- exact replica of DynamicsNetwork from ez_dmc_state.py
# ---------------------------------------------------------------------------


class DMCDynamicsBaseline(nn.Module):
    """Exact copy of the MLP-based DynamicsNetwork for DMC.

    Forward signature: (hidden [B, 128], action [B, A], reward_hidden=None) -> [B, 128]
    """

    def __init__(
        self,
        hidden_shape: int = 128,
        action_shape: int = 4,
        num_blocks: int = 2,
        dyn_shape: int = 256,
        act_embed_shape: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.hidden_shape = hidden_shape

        self.act_linear1 = nn.Linear(action_shape, act_embed_shape)
        self.act_ln1 = nn.LayerNorm(act_embed_shape)

        self.dyn_ln_1 = nn.LayerNorm(hidden_shape + act_embed_shape)
        self.dyn_net_1 = nn.Linear(hidden_shape + act_embed_shape, dyn_shape)

        self.dyn_ln_2 = nn.LayerNorm(dyn_shape)
        self.dyn_net_2 = nn.Linear(dyn_shape, hidden_shape)

        if num_blocks > 0:
            self.dyn_resblocks = nn.ModuleList(
                [ImproveResidualBlock(hidden_shape, dyn_shape) for _ in range(num_blocks)]
            )
        else:
            self.dyn_resblocks = nn.ModuleList([])

    def forward(self, hidden: torch.Tensor, action: torch.Tensor, reward_hidden=None) -> torch.Tensor:
        # Action embedding
        act_emb = F.relu(self.act_ln1(self.act_linear1(action)))

        # Dynamics MLP with residual
        x = F.relu(self.dyn_net_1(self.dyn_ln_1(torch.cat((hidden, act_emb), dim=-1))))
        x = self.dyn_net_2(x)
        state = hidden + x

        # Residual tower
        for block in self.dyn_resblocks:
            state = block(state)

        return state


# ---------------------------------------------------------------------------
# 2. Baseline + output LayerNorm (fair comparison baseline)
# ---------------------------------------------------------------------------


class DMCDynamicsBaselineNorm(nn.Module):
    """Baseline dynamics with an additional output LayerNorm.

    This is the fair baseline for sample-efficiency comparisons, analogous to
    the GroupNorm variant used in the Atari baseline.

    Forward signature: (hidden [B, 128], action [B, A], reward_hidden=None) -> [B, 128]
    """

    def __init__(
        self,
        hidden_shape: int = 128,
        action_shape: int = 4,
        num_blocks: int = 2,
        dyn_shape: int = 256,
        act_embed_shape: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.hidden_shape = hidden_shape

        self.act_linear1 = nn.Linear(action_shape, act_embed_shape)
        self.act_ln1 = nn.LayerNorm(act_embed_shape)

        self.dyn_ln_1 = nn.LayerNorm(hidden_shape + act_embed_shape)
        self.dyn_net_1 = nn.Linear(hidden_shape + act_embed_shape, dyn_shape)

        self.dyn_ln_2 = nn.LayerNorm(dyn_shape)
        self.dyn_net_2 = nn.Linear(dyn_shape, hidden_shape)

        if num_blocks > 0:
            self.dyn_resblocks = nn.ModuleList(
                [ImproveResidualBlock(hidden_shape, dyn_shape) for _ in range(num_blocks)]
            )
        else:
            self.dyn_resblocks = nn.ModuleList([])

        # Additional output normalization
        self.output_ln = nn.LayerNorm(hidden_shape)

    def forward(self, hidden: torch.Tensor, action: torch.Tensor, reward_hidden=None) -> torch.Tensor:
        # Action embedding
        act_emb = F.relu(self.act_ln1(self.act_linear1(action)))

        # Dynamics MLP with residual
        x = F.relu(self.dyn_net_1(self.dyn_ln_1(torch.cat((hidden, act_emb), dim=-1))))
        x = self.dyn_net_2(x)
        state = hidden + x

        # Residual tower
        for block in self.dyn_resblocks:
            state = block(state)

        # Output normalization
        state = self.output_ln(state)
        return state


# ---------------------------------------------------------------------------
# 3. Feature STU -- STU as a feature mixer on flat vectors
# ---------------------------------------------------------------------------


class DMCDynamicsWithSTU(nn.Module):
    """Feature STU variant: applies STU as a feature mixer over the latent vector.

    After the standard dynamics MLP + residual, the flat [B, 128] state is
    reshaped to [B, L, D] (where L * D = hidden_shape), processed by a MiniSTU
    layer that mixes across the L "pseudo-sequence" positions, reshaped back,
    and added as a second residual.  A LayerNorm is applied on output.

    Forward signature: (hidden [B, 128], action [B, A], reward_hidden=None) -> [B, 128]
    """

    def __init__(
        self,
        hidden_shape: int = 128,
        action_shape: int = 4,
        num_blocks: int = 1,
        dyn_shape: int = 256,
        act_embed_shape: int = 64,
        stu_seq_len: int = 16,
        stu_num_filters: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.hidden_shape = hidden_shape
        self.stu_seq_len = stu_seq_len
        assert hidden_shape % stu_seq_len == 0, (
            f"hidden_shape ({hidden_shape}) must be divisible by stu_seq_len ({stu_seq_len})"
        )
        self.stu_feature_dim = hidden_shape // stu_seq_len

        # --- Standard dynamics MLP (same as baseline) ---
        self.act_linear1 = nn.Linear(action_shape, act_embed_shape)
        self.act_ln1 = nn.LayerNorm(act_embed_shape)

        self.dyn_ln_1 = nn.LayerNorm(hidden_shape + act_embed_shape)
        self.dyn_net_1 = nn.Linear(hidden_shape + act_embed_shape, dyn_shape)

        self.dyn_ln_2 = nn.LayerNorm(dyn_shape)
        self.dyn_net_2 = nn.Linear(dyn_shape, hidden_shape)

        # --- Feature STU ---
        self.stu = MiniSTU(
            seq_len=stu_seq_len,
            num_filters=stu_num_filters,
            input_dim=self.stu_feature_dim,
            output_dim=self.stu_feature_dim,
        )

        # --- Output norm ---
        self.output_ln = nn.LayerNorm(hidden_shape)

        # --- Residual blocks (fewer since STU adds capacity) ---
        if num_blocks > 0:
            self.dyn_resblocks = nn.ModuleList(
                [ImproveResidualBlock(hidden_shape, dyn_shape) for _ in range(num_blocks)]
            )
        else:
            self.dyn_resblocks = nn.ModuleList([])

    def forward(self, hidden: torch.Tensor, action: torch.Tensor, reward_hidden=None) -> torch.Tensor:
        B = hidden.shape[0]

        # Action embedding
        act_emb = F.relu(self.act_ln1(self.act_linear1(action)))

        # Dynamics MLP with residual
        x = F.relu(self.dyn_net_1(self.dyn_ln_1(torch.cat((hidden, act_emb), dim=-1))))
        x = self.dyn_net_2(x)
        state = hidden + x  # first residual

        # Feature STU: reshape -> STU -> reshape -> residual
        stu_in = state.view(B, self.stu_seq_len, self.stu_feature_dim)  # [B, L, D]
        stu_out = self.stu(stu_in)  # [B, L, D]
        stu_out = stu_out.reshape(B, self.hidden_shape)  # [B, hidden_shape]
        state = state + stu_out  # second residual

        # Output normalization
        state = self.output_ln(state)

        # Residual tower
        for block in self.dyn_resblocks:
            state = block(state)

        return state


# ---------------------------------------------------------------------------
# 4. Temporal STU -- STU over a history buffer of past states
# ---------------------------------------------------------------------------


class DMCDynamicsSTUSequential(nn.Module):
    """Temporal STU variant: applies stacked STU layers over a history of states.

    Instead of operating on a single state, this model receives a buffer of the
    N most recent hidden states [B, N, hidden_shape], encodes each into a
    sequence representation, runs stacked STU layers over the temporal dimension,
    then decodes (last position + action embedding) back to a single next state.

    Forward signature:
        (hidden_history [B, N, 128], action [B, A], reward_hidden=None) -> [B, 128]

    The caller is responsible for maintaining and updating the history buffer.
    """

    def __init__(
        self,
        hidden_shape: int = 128,
        action_shape: int = 4,
        num_blocks: int = 0,
        dyn_shape: int = 256,
        act_embed_shape: int = 64,
        buffer_size: int = 10,
        seq_hidden_dim: int = 256,
        stu_num_filters: int = 2,
        num_stu_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.hidden_shape = hidden_shape
        self.buffer_size = buffer_size
        self.seq_hidden_dim = seq_hidden_dim

        # --- State encoder ---
        self.state_enc_linear1 = nn.Linear(hidden_shape, seq_hidden_dim)
        self.state_enc_ln1 = nn.LayerNorm(seq_hidden_dim)
        self.state_enc_linear2 = nn.Linear(seq_hidden_dim, seq_hidden_dim)
        self.state_enc_ln2 = nn.LayerNorm(seq_hidden_dim)

        # --- Stacked STU layers with residual + LayerNorm ---
        self.stu_layers = nn.ModuleList()
        self.stu_layer_norms = nn.ModuleList()
        for _ in range(num_stu_layers):
            self.stu_layers.append(
                MiniSTU(
                    seq_len=buffer_size,
                    num_filters=stu_num_filters,
                    input_dim=seq_hidden_dim,
                    output_dim=seq_hidden_dim,
                )
            )
            self.stu_layer_norms.append(nn.LayerNorm(seq_hidden_dim))

        # --- Action embedding (continuous actions, so Linear not Embedding) ---
        self.act_linear = nn.Linear(action_shape, seq_hidden_dim)
        self.act_ln = nn.LayerNorm(seq_hidden_dim)

        # --- Decoder: concat last STU output with action embedding ---
        self.decoder = nn.Linear(seq_hidden_dim + seq_hidden_dim, hidden_shape)

        # --- Output LayerNorm ---
        self.output_ln = nn.LayerNorm(hidden_shape)

    def forward(
        self,
        hidden_history: torch.Tensor,
        action: torch.Tensor,
        reward_hidden=None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_history : Tensor [B, N, hidden_shape]
            Buffer of the N most recent hidden states (oldest first).
        action : Tensor [B, action_shape]
            Current action.

        Returns
        -------
        next_state : Tensor [B, hidden_shape]
        """
        B, N, H = hidden_history.shape

        # The most recent state is used for the residual connection
        current_hidden = hidden_history[:, -1, :]  # [B, hidden_shape]

        # Encode each state in the history
        x = self.state_enc_linear1(hidden_history)  # [B, N, seq_hidden_dim]
        x = self.state_enc_ln1(x)
        x = F.relu(x)
        x = self.state_enc_linear2(x)  # [B, N, seq_hidden_dim]
        x = self.state_enc_ln2(x)

        # Stacked STU layers with residual + LayerNorm
        for stu_layer, ln in zip(self.stu_layers, self.stu_layer_norms):
            residual = x
            x = stu_layer(x)  # [B, N, seq_hidden_dim]
            x = x + residual
            x = ln(x)

        # Take the last position as the temporal summary
        x_last = x[:, -1, :]  # [B, seq_hidden_dim]

        # Action embedding
        act_emb = F.relu(self.act_ln(self.act_linear(action)))  # [B, seq_hidden_dim]

        # Decode
        combined = torch.cat((x_last, act_emb), dim=-1)  # [B, 2 * seq_hidden_dim]
        out = self.decoder(combined)  # [B, hidden_shape]

        # Residual from input
        out = out + current_hidden

        # Output normalization
        out = self.output_ln(out)

        return out
