from functools import partial

import jax
import jax.numpy as jnp
import optax

from diffuser.diffusion import ValueDiffusion
from utilities.flax_utils import TrainState, apply_ema_decay, copy_params_to_ema
from utilities.jax_utils import next_rng, value_and_multi_grad

from .base_algo import Algo


class DiffuserValue(Algo):
    def __init__(self, cfg, value_function):
        self.config = cfg
        self.value_function = value_function
        self.horizon = self.config.horizon
        self.history_horizon = self.config.history_horizon

        self.observation_dim = value_function.sample_dim - value_function.action_dim
        self.action_dim = value_function.action_dim
        self.diffusion: ValueDiffusion = self.value_function.diffusion

        self._total_steps = 0
        self._train_states = {}

        def get_optimizer():
            opt = optax.adam(self.config.lr)
            return opt

        value_function_params = self.value_function.init(
            next_rng(),
            next_rng(),
            samples=jnp.zeros(
                (
                    10,
                    self.horizon + self.history_horizon,
                    self.value_function.sample_dim,
                )
            ),
            conditions={
                (0, self.history_horizon + 1): jnp.zeros(
                    (10, self.history_horizon + 1, self.observation_dim)
                )
            },
            ts=jnp.zeros((10,), dtype=jnp.int32),
            targets=jnp.zeros((10, 1)),
            method=self.value_function.loss,
        )
        self._train_states["value_function"] = TrainState.create(
            params=value_function_params,
            params_ema=value_function_params,
            tx=get_optimizer(),
            apply_fn=None,
        )
        self._model_keys = tuple(["value_function"])

    @partial(jax.jit, static_argnames=("self"))
    def _train_step(self, train_states, rng, batch):
        diff_loss_fn = self.get_diff_loss(batch)

        params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_value_function), grad_value_function = value_and_multi_grad(
            diff_loss_fn, 1, has_aux=True
        )(params, rng)

        train_states["value_function"] = train_states["value_function"].apply_gradients(
            grads=grad_value_function[0]["value_function"]
        )
        metrics = dict(value_loss=aux_value_function["loss"])
        return train_states, metrics

    def get_diff_loss(self, batch):
        def diff_loss(params, rng):
            samples = jnp.concatenate((batch["samples"], batch["actions"]), axis=-1)
            conditions, targets = batch["conditions"], batch["targets"]
            terms = self.get_diff_terms(params, samples, conditions, targets, rng)
            loss = terms["loss"].mean()
            return (loss,), locals()

        return diff_loss

    def get_diff_terms(self, params, samples, conditions, targets, rng):
        rng, split_rng = jax.random.split(rng)
        ts = jax.random.randint(
            split_rng,
            (samples.shape[0],),
            minval=0,
            maxval=self.diffusion.num_timesteps,
        )
        rng, split_rng = jax.random.split(rng)
        terms = self.value_function.apply(
            params["value_function"],
            split_rng,
            samples,
            conditions,
            targets,
            ts,
            method=self.value_function.loss,
        )
        return terms

    def train(self, batch):
        self._total_steps += 1
        self._train_states, metrics = self._train_step(
            self._train_states, next_rng(), batch
        )
        if self._total_steps % self.config.update_ema_every == 0:
            self.step_ema()
        return metrics

    def step_ema(self):
        if self._total_steps < self.config.step_start_ema:
            self._train_states["value_function"] = copy_params_to_ema(
                self._train_states["value_function"]
            )
        else:
            self._train_states["value_function"] = apply_ema_decay(
                self._train_states["value_function"], self.config.ema_decay
            )

    @property
    def model_keys(self):
        return self._model_keys

    @property
    def train_states(self):
        return self._train_states

    @property
    def train_params(self):
        return {key: self.train_states[key].params for key in self.model_keys}

    @property
    def eval_params(self):
        return {
            key: self.train_states[key].params_ema or self.train_states[key].params
            for key in self.model_keys
        }

    @property
    def total_steps(self):
        return self._total_steps
