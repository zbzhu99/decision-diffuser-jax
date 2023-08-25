from functools import partial

import jax
import jax.numpy as jnp

from utilities.jax_utils import next_rng


class SamplerPolicy(object):  # used for dql
    def __init__(
        self, policy, qf=None, mean=0, std=1, ensemble=False, act_method="ddpm"
    ):
        self.policy = policy
        self.qf = qf
        self.mean = mean
        self.std = std
        self.num_samples = 50
        self.act_method = act_method

    def update_params(self, params):
        self.params = params
        return self

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def act(self, params, rng, observations, deterministic):
        conditions = {}
        return self.policy.apply(
            params["policy"], rng, observations, conditions, deterministic, repeat=None
        )

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def ensemble_act(self, params, rng, observations, deterministic, num_samples):
        rng, key = jax.random.split(rng)
        conditions = {}
        actions = self.policy.apply(
            params["policy"],
            key,
            observations,
            conditions,
            deterministic,
            repeat=num_samples,
        )
        q1 = self.qf.apply(params["qf1"], observations, actions)
        q2 = self.qf.apply(params["qf2"], observations, actions)
        q = jnp.minimum(q1, q2)

        idx = jax.random.categorical(rng, q)
        return actions[jnp.arange(actions.shape[0]), idx]

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def ddpmensemble_act(self, params, rng, observations, deterministic, num_samples):
        rng, key = jax.random.split(rng)
        conditions = {}
        actions = self.policy.apply(
            params["policy"],
            rng,
            observations,
            conditions,
            deterministic,
            repeat=num_samples,
            method=self.policy.ddpm_sample,
        )
        q1 = self.qf.apply(params["qf1"], observations, actions)
        q2 = self.qf.apply(params["qf2"], observations, actions)
        q = jnp.minimum(q1, q2)

        idx = jax.random.categorical(rng, q)
        return actions[jnp.arange(actions.shape[0]), idx]

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def dpmensemble_act(self, params, rng, observations, deterministic, num_samples):
        rng, key = jax.random.split(rng)
        conditions = {}
        actions = self.policy.apply(
            params["policy"],
            rng,
            observations,
            conditions,
            deterministic,
            repeat=num_samples,
            method=self.policy.dpm_sample,
        )
        q1 = self.qf.apply(params["qf1"], observations, actions)
        q2 = self.qf.apply(params["qf2"], observations, actions)
        q = jnp.minimum(q1, q2)

        idx = jax.random.categorical(rng, q)
        return actions[jnp.arange(actions.shape[0]), idx]

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def dpm_act(self, params, rng, observations, deterministic, num_samples):
        conditions = {}
        return self.policy.apply(
            params["policy"],
            rng,
            observations,
            conditions,
            deterministic,
            method=self.policy.dpm_sample,
        )

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def ddim_act(self, params, rng, observations, deterministic, num_samples):
        conditions = {}
        return self.policy.apply(
            params["policy"],
            rng,
            observations,
            conditions,
            deterministic,
            method=self.policy.ddim_sample,
        )

    @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
    def ddpm_act(self, params, rng, observations, deterministic, num_samples):
        conditions = {}
        return self.policy.apply(
            params["policy"],
            rng,
            observations,
            conditions,
            deterministic,
            method=self.policy.ddpm_sample,
        )

    def __call__(self, observations, deterministic=False):
        actions = getattr(self, f"{self.act_method}_act")(
            self.params, next_rng(), observations, deterministic, self.num_samples
        )
        if isinstance(actions, tuple):
            actions = actions[0]
        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)


class DiffuserPolicy(object):
    def __init__(self, planner, inv_model, act_method="ddpm"):
        self.planner = planner
        self.inv_model = inv_model
        self.act_method = act_method

    def update_params(self, params):
        self.params = params
        return self

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def ddpm_act(
        self, params, rng, observations, deterministic
    ):  # deterministic is not used
        conditions = {0: observations}
        returns = jnp.ones((observations.shape[0], 1)) * 0.9
        plan_samples = self.planner.apply(
            params["planner"],
            rng,
            conditions=conditions,
            returns=returns,
            method=self.planner.ddpm_sample,
        )

        if self.inv_model is not None:
            obs_comb = jnp.concatenate(
                [plan_samples[:, 0], plan_samples[:, 1]], axis=-1
            )
            actions = self.inv_model.apply(
                params["inv_model"],
                obs_comb,
            )
        else:
            actions = plan_samples[:, 0, -self.planner.action_dim :]

        return actions

    @partial(jax.jit, static_argnames=("self", "deterministic"))
    def ddim_act(
        self, params, rng, observations, deterministic
    ):  # deterministic is not used
        conditions = {0: observations}
        returns = jnp.ones((observations.shape[0], 1)) * 0.9
        plan_samples = self.planner.apply(
            params["planner"],
            rng,
            conditions=conditions,
            returns=returns,
            method=self.planner.ddim_sample,
        )

        if self.inv_model is not None:
            obs_comb = jnp.concatenate(
                [plan_samples[:, 0], plan_samples[:, 1]], axis=-1
            )
            actions = self.inv_model.apply(
                params["inv_model"],
                obs_comb,
            )
        else:
            actions = plan_samples[:, 0, -self.planner.action_dim :]

        return actions

    def __call__(self, observations, deterministic=False):
        actions = getattr(self, f"{self.act_method}_act")(
            self.params, next_rng(), observations, deterministic
        )
        assert jnp.all(jnp.isfinite(actions))
        return jax.device_get(actions)
