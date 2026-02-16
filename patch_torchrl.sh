#!/bin/bash
set -e

PROJ="/lambda/nfs/Jan-21-2026/STUZero"

# 1. Create the compatibility module
cat > "$PROJ/ez/utils/torchrl_compat.py" << 'PYEOF'
"""Drop-in replacements for the torchrl modules used in EfficientZero-v2."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class NoisyLinear(nn.Module):
    """Noisy linear layer (Fortunato et al., 2018)."""

    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon,
            )
        return F.linear(x, self.weight_mu, self.bias_mu)


class TruncatedNormal(torch.distributions.Distribution):
    """Truncated normal distribution clamped to [low, high]."""

    has_rsample = True

    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        self.loc = loc
        self.scale = scale
        self.low = low
        self.high = high
        self.eps = eps
        self._normal = Normal(torch.zeros_like(loc), torch.ones_like(scale))
        super().__init__(batch_shape=loc.shape, validate_args=False)

    def _clamp(self, x):
        return torch.clamp(x, self.low + self.eps, self.high - self.eps)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        x = torch.normal(
            self.loc.expand(shape), self.scale.expand(shape)
        )
        return self._clamp(x)

    def log_prob(self, value):
        return self._normal.log_prob(
            (value - self.loc) / self.scale
        ) - torch.log(self.scale)
PYEOF

echo "Created $PROJ/ez/utils/torchrl_compat.py"

# 2. Patch ez/utils/loss.py
sed -i 's/^import torchrl$/from ez.utils.torchrl_compat import TruncatedNormal as _TruncatedNormal/' "$PROJ/ez/utils/loss.py"
sed -i 's/torchrl\.modules\.TruncatedNormal(/_TruncatedNormal(/g' "$PROJ/ez/utils/loss.py"
echo "Patched $PROJ/ez/utils/loss.py"

# 3. Patch ez/agents/ez_dmc_state.py
sed -i 's/^import torchrl$/from ez.utils.torchrl_compat import NoisyLinear/' "$PROJ/ez/agents/ez_dmc_state.py"
sed -i 's/torchrl\.modules\.NoisyLinear(/NoisyLinear(/g' "$PROJ/ez/agents/ez_dmc_state.py"
echo "Patched $PROJ/ez/agents/ez_dmc_state.py"

# 4. Patch ez/mcts/cy_mcts.py — remove bare import
sed -i '/^import torchrl$/d' "$PROJ/ez/mcts/cy_mcts.py"
echo "Patched $PROJ/ez/mcts/cy_mcts.py"

# 5. Remove torchrl and tensordict
pip uninstall -y torchrl tensordict 2>/dev/null || true
echo ""
echo "Done! torchrl dependency has been replaced with local implementations."