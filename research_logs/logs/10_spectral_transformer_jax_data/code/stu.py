from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx

from stj.config import ModelConfig

from .common import Dense


def nearest_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << int(math.ceil(math.log2(n)))


def get_hankel(seq_len: int, use_hankel_L: bool = False) -> jax.Array:
    entries = jnp.arange(1, seq_len + 1, dtype=jnp.float32)
    i_plus_j = entries[:, None] + entries[None, :]
    if use_hankel_L:
        sign = jnp.power(-1.0, i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        return sign * (8.0 / denom)
    return 2.0 / (i_plus_j**3 - i_plus_j)


def get_spectral_filters(
    seq_len: int,
    num_eigh: int,
    use_hankel_L: bool = False,
    random: bool = False,
    random_normalized: bool = False,
    seed: int = 0,
) -> jax.Array:
    if random:
        key = jax.random.PRNGKey(seed)
        if random_normalized:
            # Haar-random orthonormal columns with the SAME column-energy spectrum
            # as the Hankel eigenbasis (sigma^(1/4) scaling). Isolates "is it the
            # Z-eigenvectors specifically that matter, vs random orthonormal
            # directions with the same energy profile?"
            g = jax.random.normal(key, (seq_len, num_eigh), dtype=jnp.float32)
            q, r = jnp.linalg.qr(g)
            q = q * jnp.sign(jnp.diag(r))  # de-bias QR -> true Haar measure
            hankel = get_hankel(seq_len, use_hankel_L=use_hankel_L)
            sigma, _ = jnp.linalg.eigh(hankel)
            sigma = sigma[-num_eigh:]
            return (q * jnp.power(jnp.maximum(sigma, 1e-8), 0.25)).astype(jnp.float32)
        return jax.random.normal(key, (seq_len, num_eigh), dtype=jnp.float32)
    hankel = get_hankel(seq_len, use_hankel_L=use_hankel_L)
    sigma, phi = jnp.linalg.eigh(hankel)
    sigma = sigma[-num_eigh:]
    phi = phi[:, -num_eigh:]
    phi = phi * jnp.power(jnp.maximum(sigma, 1e-8), 0.25)
    return phi.astype(jnp.float32)


def fft_convolve(x: jax.Array, filters: jax.Array) -> jax.Array:
    seq_len = x.shape[1]
    fft_size = nearest_power_of_two(2 * seq_len - 1)
    x_f = jnp.fft.rfft(x.astype(jnp.float32), n=fft_size, axis=1)
    f_f = jnp.fft.rfft(filters.astype(jnp.float32), n=fft_size, axis=0)
    y_f = x_f * f_f[None, :, :]
    y = jnp.fft.irfft(y_f, n=fft_size, axis=1)
    return y[:, :seq_len, :]


def fft_convolve_basis(x: jax.Array, phi: jax.Array) -> jax.Array:
    seq_len = x.shape[1]
    fft_size = nearest_power_of_two(2 * seq_len - 1)
    x_f = jnp.fft.rfft(x.astype(jnp.float32), n=fft_size, axis=1)
    phi_f = jnp.fft.rfft(phi.astype(jnp.float32), n=fft_size, axis=0)
    y_f = x_f[:, :, None, :] * phi_f[None, :, :, None]
    y = jnp.fft.irfft(y_f, n=fft_size, axis=1)
    return y[:, :seq_len, :, :]


class STUMixer(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs, branch_dim: int | None = None):
        self.config = config
        self._use_branch = branch_dim is not None
        self.branch_dim = branch_dim if branch_dim is not None else config.d_model
        self.phi = get_spectral_filters(
            config.max_sequence_length,
            config.stu_num_eigh,
            use_hankel_L=config.stu_use_hankel_L,
            random=config.stu_random_filters,
            random_normalized=config.stu_random_normalized,
        )
        if config.stu_use_approx:
            self.input_proj = Dense(config.d_model, self.branch_dim, use_bias=False, rngs=rngs)
            self.filter_proj = nnx.Param(
                jax.random.normal(rngs.params(), (config.stu_num_eigh, self.branch_dim)) / math.sqrt(self.branch_dim)
            )
            self.full_mix = None
        else:
            if self._use_branch:
                self.input_proj = Dense(config.d_model, self.branch_dim, use_bias=False, rngs=rngs)
            else:
                self.input_proj = None
            self.filter_proj = None
            self.full_mix = nnx.Param(
                jax.random.normal(
                    rngs.params(), (config.stu_num_eigh, self.branch_dim, self.branch_dim)
                )
                / math.sqrt(self.branch_dim)
            )
        self.out_proj = Dense(self.branch_dim, self.branch_dim, use_bias=config.use_bias, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        seq_len = x.shape[1]
        phi = self.phi[:seq_len]
        if self.config.stu_use_approx:
            inputs = self.input_proj(x)
            filters = jnp.einsum("tk,kd->td", phi, self.filter_proj)
            mixed = fft_convolve(inputs, filters)
        else:
            inputs = self.input_proj(x) if self.input_proj is not None else x
            basis = fft_convolve_basis(inputs, phi)
            mixed = jnp.einsum("btkc,kco->bto", basis, self.full_mix)
        return self.out_proj(mixed.astype(x.dtype))

