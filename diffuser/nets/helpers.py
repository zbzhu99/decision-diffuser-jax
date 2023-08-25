import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from utilities.jax_utils import extend_and_repeat


class Conv1dBlock(nn.Module):
    out_channels: int
    kernel_size: int
    mish: bool = True
    n_groups: int = 8

    @nn.compact
    def __call__(self, x):
        if self.mish:
            act_fn = mish
        else:
            act_fn = nn.silu

        # NOTE(zbzhu): in flax, conv use the channel last format
        x = nn.Conv(
            self.out_channels, (self.kernel_size,), padding=self.kernel_size // 2
        )(x)
        x = nn.GroupNorm(self.n_groups, epsilon=1e-5)(x)
        return act_fn(x)


class DownSample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.dim, (3,), strides=(2,), padding=1)(x)
        return x


class UpSample1d(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        batch, length, channels = x.shape
        x = jax.image.resize(
            x,
            shape=(batch, length * 2, channels),
            method="nearest",
        )
        x = nn.Conv(self.dim, (3,), strides=(1,), padding=1)(x)
        return x


class GaussianPolicy(nn.Module):
    action_dim: int
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    temperature: float = 1.0

    @nn.compact
    def __call__(self, mean):
        log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        return distrax.MultivariateNormalDiag(
            mean, jnp.exp(log_stds * self.temperature)
        )


def multiple_action_q_function(forward):
    def wrapped(self, observations, actions, **kwargs):
        multiple_actions = False
        batch_size = observations.shape[0]
        if actions.ndim == 3 and observations.ndim == 2:
            multiple_actions = True
            observations = extend_and_repeat(observations, 1, actions.shape[1])
            observations = observations.reshape(-1, observations.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
        q_values = forward(self, observations, actions, **kwargs)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values

    return wrapped


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


def sinusoidal_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = jnp.exp(-jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half)
    # args = timesteps[:, None] * freqs[None, :]
    args = jnp.expand_dims(timesteps, axis=-1) * freqs[None, :]
    embd = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
    return embd


class TimeEmbedding(nn.Module):
    embed_size: int
    act: callable = mish

    @nn.compact
    def __call__(self, timesteps):
        x = sinusoidal_embedding(timesteps, self.embed_size)
        x = nn.Dense(self.embed_size * 4)(x)
        x = self.act(x)
        x = nn.Dense(self.embed_size)(x)
        return x
