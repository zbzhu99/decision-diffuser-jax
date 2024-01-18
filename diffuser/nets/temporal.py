from typing import Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from einops.layers.flax import Rearrange

from diffuser.diffusion import GaussianDiffusion, ValueDiffusion
from diffuser.nets.helpers import (
    Conv1dBlock,
    DownSample1d,
    TimeEmbedding,
    UpSample1d,
    mish,
)


class ResidualTemporalBlock(nn.Module):
    out_channels: int
    kernel_size: int
    mish: bool = True

    @nn.compact
    def __call__(self, x, t):
        if self.mish:
            act_fn = mish
        else:
            act_fn = nn.silu

        time_mlp = nn.Sequential(
            [
                act_fn,
                nn.Dense(self.out_channels),
                Rearrange("batch f -> batch 1 f"),
            ]
        )

        out = Conv1dBlock(self.out_channels, self.kernel_size, self.mish)(x) + time_mlp(
            t
        )
        out = Conv1dBlock(self.out_channels, self.kernel_size, self.mish)(out)

        if x.shape[-1] == self.out_channels:
            return out
        else:
            return out + nn.Conv(self.out_channels, (1,))(x)


class TemporalUnet(nn.Module):
    sample_dim: int
    dim: int = 128
    dim_mults: Tuple[int] = (1, 4, 8)
    env_ts_condition: bool = False
    returns_condition: bool = False
    condition_dropout: float = 0.25
    kernel_size: int = 5
    max_traj_length: int = 1000

    def setup(self):
        self.dims = dims = [
            self.sample_dim,
            *map(lambda m: self.dim * m, self.dim_mults),
        ]
        self.in_out = list(zip(dims[:-1], dims[1:]))

    @nn.compact
    def __call__(
        self,
        rng,
        x: jnp.ndarray,
        time: jnp.ndarray,
        env_ts: jnp.ndarray = None,
        returns_to_go: jnp.ndarray = None,
        use_dropout: bool = True,
        force_dropout: bool = False,
    ):
        act_fn = mish

        emb = TimeEmbedding(self.dim)(time)
        if self.returns_condition or self.env_ts_condition:
            emb = jnp.expand_dims(emb, 1)

        if self.env_ts_condition:
            env_ts_emb = nn.Embed(self.max_traj_length, self.dim)(env_ts)
            emb = jnp.concatenate([emb, jnp.expand_dims(env_ts_emb, 1)], axis=1)

        if self.returns_condition:
            assert returns_to_go is not None
            returns_to_go = returns_to_go.reshape(-1, 1)
            returns_emb = nn.Sequential(
                [
                    nn.Dense(self.dim),
                    act_fn,
                    nn.Dense(self.dim * 4),
                    act_fn,
                    nn.Dense(self.dim),
                ]
            )(returns_to_go)

            mask_dist = distrax.Bernoulli(probs=1 - self.condition_dropout)
            if use_dropout:
                rng, sample_key = jax.random.split(rng)
                mask = mask_dist.sample(
                    seed=sample_key, sample_shape=(returns_emb.shape[0], 1)
                )
                returns_emb = returns_emb * mask
            if force_dropout:
                returns_emb = returns_emb * 0
            emb = jnp.concatenate([emb, jnp.expand_dims(returns_emb, 1)], axis=1)

        if self.returns_condition or self.env_ts_condition:
            emb = nn.LayerNorm()(emb)
            emb = emb.reshape(-1, emb.shape[1] * emb.shape[2])

        h = []
        num_resolutions = len(self.in_out)
        for ind, (_, dim_out) in enumerate(self.in_out):
            is_last = ind >= (num_resolutions - 1)

            x = ResidualTemporalBlock(
                dim_out,
                kernel_size=self.kernel_size,
                mish=True,
            )(x, emb)
            x = ResidualTemporalBlock(
                dim_out,
                kernel_size=self.kernel_size,
                mish=True,
            )(x, emb)
            h.append(x)

            if not is_last:
                x = DownSample1d(dim_out)(x)

        mid_dim = self.dims[-1]
        x = ResidualTemporalBlock(
            mid_dim,
            kernel_size=self.kernel_size,
            mish=True,
        )(x, emb)
        x = ResidualTemporalBlock(
            mid_dim,
            kernel_size=self.kernel_size,
            mish=True,
        )(x, emb)

        for ind, (dim_in, _) in enumerate(reversed(self.in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            x = jnp.concatenate([x, h.pop()], axis=-1)
            x = ResidualTemporalBlock(
                dim_in,
                kernel_size=self.kernel_size,
                mish=True,
            )(x, emb)
            x = ResidualTemporalBlock(
                dim_in,
                kernel_size=self.kernel_size,
                mish=True,
            )(x, emb)

            if not is_last:
                x = UpSample1d(dim_in)(x)

        x = nn.Sequential(
            [
                Conv1dBlock(self.dim, kernel_size=self.kernel_size, mish=True),
                nn.Conv(self.sample_dim, (1,)),
            ]
        )(x)

        return x


class DiffusionPlanner(nn.Module):
    diffusion: GaussianDiffusion
    sample_dim: int
    action_dim: int
    horizon: int
    history_horizon: int
    dim: int
    dim_mults: Tuple[int]
    env_ts_condition: bool = True
    returns_condition: bool = True
    condition_dropout: float = 0.25
    kernel_size: int = 5
    sample_method: str = "ddpm"
    dpm_steps: int = 15
    dpm_t_end: float = 0.001
    max_traj_length: int = 1000

    def setup(self):
        self.base_net = TemporalUnet(
            sample_dim=self.sample_dim,
            dim=self.dim,
            dim_mults=self.dim_mults,
            env_ts_condition=self.env_ts_condition,
            returns_condition=self.returns_condition,
            condition_dropout=self.condition_dropout,
            kernel_size=self.kernel_size,
            max_traj_length=self.max_traj_length,
        )

    def ddpm_sample(
        self, rng, conditions, env_ts=None, deterministic=False, returns_to_go=None
    ):
        batch_size = list(conditions.values())[0].shape[0]
        return self.diffusion.p_sample_loop_jit(
            rng_key=rng,
            model_forward=self.base_net,
            shape=(batch_size, self.horizon + self.history_horizon, self.sample_dim),
            conditions=conditions,
            condition_dim=self.sample_dim - self.action_dim,
            returns_to_go=returns_to_go,
            env_ts=env_ts,
        )

    def __call__(
        self, rng, conditions, env_ts=None, deterministic=False, returns_to_go=None
    ):
        return getattr(self, f"{self.sample_method}_sample")(
            rng, conditions, env_ts, deterministic, returns_to_go
        )

    def loss(
        self,
        rng_key,
        samples,
        conditions,
        ts,
        env_ts=None,
        masks=None,
        returns_to_go=None,
    ):
        terms = self.diffusion.training_losses(
            rng_key,
            model_forward=self.base_net,
            x_start=samples,
            conditions=conditions,
            condition_dim=self.sample_dim - self.action_dim,
            returns_to_go=returns_to_go,
            env_ts=env_ts,
            t=ts,
            masks=masks,
        )
        return terms


class ValueNet(nn.Module):
    sample_dim: int
    dim: int = 32
    dim_mults: Tuple[int] = (1, 2, 4, 8)
    kernel_size: int = 5

    def setup(self):
        self.dims = dims = [
            self.sample_dim,
            *map(lambda m: self.dim * m, self.dim_mults),
        ]
        self.in_out = list(zip(dims[:-1], dims[1:]))

    @nn.compact
    def __call__(
        self,
        rng,
        x: jnp.ndarray,
        time: jnp.ndarray,
    ):
        emb = TimeEmbedding(self.dim)(time)
        horizon = x.shape[1]

        h = []
        num_resolutions = len(self.in_out)
        for ind, (_, dim_out) in enumerate(self.in_out):
            is_last = ind >= (num_resolutions - 1)

            x = ResidualTemporalBlock(
                dim_out,
                kernel_size=self.kernel_size,
                mish=True,
            )(x, emb)
            x = ResidualTemporalBlock(
                dim_out,
                kernel_size=self.kernel_size,
                mish=True,
            )(x, emb)
            h.append(x)

            if not is_last:
                x = DownSample1d(dim_out)(x)
                horizon = horizon // 2

        mid_dim = self.dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        x = ResidualTemporalBlock(
            mid_dim_2,
            kernel_size=self.kernel_size,
            mish=True,
        )(x, emb)
        x = DownSample1d(mid_dim_2)(x)
        horizon = horizon // 2
        x = ResidualTemporalBlock(
            mid_dim_3,
            kernel_size=self.kernel_size,
            mish=True,
        )(x, emb)
        x = DownSample1d(mid_dim_3)(x)
        horizon = horizon // 2

        fc_dim = mid_dim_3 * max(horizon, 1)
        x = x.reshape(len(x), -1)
        out = nn.Sequential(
            [
                nn.Dense(fc_dim // 2),
                mish,
                nn.Dense(1),
            ]
        )(jnp.concatenate([x, emb], axis=-1))
        return out


class ValueFunction(nn.Module):
    diffusion: ValueDiffusion
    sample_dim: int
    action_dim: int
    dim: int
    dim_mults: Tuple[int]
    kernel_size: int = 5

    def setup(self):
        self.value_net = ValueNet(
            sample_dim=self.sample_dim,
            dim=self.dim,
            dim_mults=self.dim_mults,
            kernel_size=self.kernel_size,
        )

    def __call__(
        self,
        rng_key,
        samples,
        conditions,
        ts,
    ):
        pred_values = self.diffusion.forward(
            rng_key,
            model_forward=self.value_net,
            x_start=samples,
            conditions=conditions,
            condition_dim=self.sample_dim - self.action_dim,
            t=ts,
        )
        return pred_values

    def loss(
        self,
        rng_key,
        samples,
        conditions,
        targets,
        ts,
    ):
        terms = self.diffusion.training_losses(
            rng_key,
            model_forward=self.value_net,
            x_start=samples,
            conditions=conditions,
            target=targets,
            condition_dim=self.sample_dim - self.action_dim,
            t=ts,
        )
        return terms
