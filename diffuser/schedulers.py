from typing import Optional, Tuple, Union
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_ddpm_flax import (
    FlaxDDPMScheduler,
    FlaxDDPMSchedulerOutput,
    DDPMSchedulerState,
)
from diffusers.schedulers.scheduling_ddim_flax import (
    FlaxDDIMScheduler,
    FlaxDDIMSchedulerOutput,
    DDIMSchedulerState,
)


@dataclass
class ModifiedFlaxDDPMSchedulerOutput(FlaxDDPMSchedulerOutput):
    pred_original_sample: jnp.ndarray


class ModifiedFlaxDDPMScheduler(FlaxDDPMScheduler):
    def step(
        self,
        state: DDPMSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        key: Optional[jax.random.KeyArray] = None,
        return_dict: bool = True,
    ) -> Union[FlaxDDPMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            state (`DDPMSchedulerState`): the `FlaxDDPMScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            key (`jax.random.KeyArray`): a PRNG key.
            return_dict (`bool`): option for returning tuple rather than FlaxDDPMSchedulerOutput class

        Returns:
            [`FlaxDDPMSchedulerOutput`] or `tuple`: [`FlaxDDPMSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        t = timestep

        if key is None:
            key = jax.random.PRNGKey(0)

        if model_output.shape[1] == sample.shape[
            1
        ] * 2 and self.config.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = jnp.split(
                model_output, sample.shape[1], axis=1
            )
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = state.common.alphas_cumprod[t]
        alpha_prod_t_prev = jnp.where(
            t > 0, state.common.alphas_cumprod[t - 1], jnp.array(1.0, dtype=self.dtype)
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` "
                " for the FlaxDDPMScheduler."
            )

        # 3. Clip "predicted x_0"
        if self.config.clip_sample:
            pred_original_sample = jnp.clip(pred_original_sample, -1, 1)

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * state.common.betas[t]
        ) / beta_prod_t
        current_sample_coeff = (
            state.common.alphas[t] ** (0.5) * beta_prod_t_prev / beta_prod_t
        )

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * sample
        )

        # 6. Add noise
        def random_variance():
            split_key = jax.random.split(key, num=1)
            noise = jax.random.normal(
                split_key, shape=model_output.shape, dtype=self.dtype
            )
            return (
                self._get_variance(state, t, predicted_variance=predicted_variance)
                ** 0.5
            ) * noise

        variance = jnp.where(
            t > 0, random_variance(), jnp.zeros(model_output.shape, dtype=self.dtype)
        )

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample, pred_original_sample, state)

        return ModifiedFlaxDDPMSchedulerOutput(
            prev_sample=pred_prev_sample,
            pred_original_sample=pred_original_sample,
            state=state,
        )


@dataclass
class ModifiedFlaxDDIMSchedulerOutput(FlaxDDIMSchedulerOutput):
    pred_original_sample: jnp.ndarray


class ModifiedFlaxDDIMScheduler(FlaxDDIMScheduler):
    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[jnp.ndarray] = None,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        clip_sample: bool = True,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.dtype = dtype

    def step(
        self,
        state: DDIMSchedulerState,
        model_output: jnp.ndarray,
        timestep: int,
        sample: jnp.ndarray,
        eta: float = 0.0,
        return_dict: bool = True,
    ) -> Union[FlaxDDIMSchedulerOutput, Tuple]:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            state (`DDIMSchedulerState`): the `FlaxDDIMScheduler` state data class instance.
            model_output (`jnp.ndarray`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`jnp.ndarray`):
                current instance of sample being created by diffusion process.
            return_dict (`bool`): option for returning tuple rather than FlaxDDIMSchedulerOutput class

        Returns:
            [`FlaxDDIMSchedulerOutput`] or `tuple`: [`FlaxDDIMSchedulerOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is the sample tensor.

        """
        if state.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = (
            timestep - self.config.num_train_timesteps // state.num_inference_steps
        )

        alphas_cumprod = state.common.alphas_cumprod
        final_alpha_cumprod = state.final_alpha_cumprod

        # 2. compute alphas, betas
        alpha_prod_t = alphas_cumprod[timestep]
        alpha_prod_t_prev = jnp.where(
            prev_timestep >= 0, alphas_cumprod[prev_timestep], final_alpha_cumprod
        )

        beta_prod_t = 1 - alpha_prod_t

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t ** (0.5) * model_output
            ) / alpha_prod_t ** (0.5)
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (
                sample - alpha_prod_t ** (0.5) * pred_original_sample
            ) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (
                beta_prod_t**0.5
            ) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (
                beta_prod_t**0.5
            ) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

        if self.config.clip_sample:
            pred_original_sample = jnp.clip(pred_original_sample, -1, 1)

        # 4. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(state, timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        # 5. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (
            0.5
        ) * pred_epsilon

        # 6. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = (
            alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
        )

        if not return_dict:
            return (prev_sample, pred_original_sample, state)

        return ModifiedFlaxDDIMSchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample,
            state=state,
        )
