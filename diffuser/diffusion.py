# Copyright 2023 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection
of beta schedules.
"""

from typing import Optional, Dict, Tuple, Callable
import enum

import flax
import jax
import jax.numpy as np

from utilities.utils import apply_conditioning
from diffuser.schedulers import ModifiedFlaxDDPMScheduler, ModifiedFlaxDDIMScheduler


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(axis=list(range(1, len(tensor.shape))))


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        num_timesteps: int,
        # TODO(zbzbu): allow linear schedule by setting `schedule_name`
        schedule_name: str,
        loss_type: str,
        env_ts_condition: bool = False,
        returns_condition: bool = False,
        condition_guidance_w: float = 1.2,
        min_value: float = -1.0,
        max_value: float = 1.0,
        rescale_timesteps: bool = False,
        sample_temperature: float = 1.0,
        use_ddim: bool = False,
        n_ddim_steps: int = 15,
    ):
        self.schedule_name = schedule_name
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps
        self.min_value = min_value
        self.max_value = max_value
        self.sample_temperature = sample_temperature
        self.loss_weights = None  # set externally
        self.use_ddim = use_ddim

        self.env_ts_condition = env_ts_condition
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        self.num_timesteps = int(num_timesteps)
        if self.use_ddim:
            self.num_inference_steps = int(n_ddim_steps)
            self.noise_scheduler = ModifiedFlaxDDIMScheduler(
                num_train_timesteps=self.num_timesteps,
                clip_sample=True,
                prediction_type="epsilon",
                beta_schedule="squaredcos_cap_v2",
            )
            noise_scheduler_state = self.noise_scheduler.create_state()
            self.noise_scheduler_state = self.noise_scheduler.set_timesteps(
                noise_scheduler_state, self.num_inference_steps
            )
        else:
            self.noise_scheduler = ModifiedFlaxDDPMScheduler(
                num_train_timesteps=self.num_timesteps,
                clip_sample=True,
                prediction_type="epsilon",
                beta_schedule="squaredcos_cap_v2",
            )
            self.noise_scheduler_state = self.noise_scheduler.create_state()
            self.num_inference_steps = self.num_timesteps

        self.betas = self.noise_scheduler_state.common.betas
        self.alphas = self.noise_scheduler_state.common.alphas
        self.alphas_cumprod = self.noise_scheduler_state.common.alphas_cumprod
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )

        self.ts_weights = ws = self.betas / (
            2 * (1 - self.alphas_cumprod) * self.alphas
        )
        self.normalized_ts_weights = ws * num_timesteps / ws.sum()

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample_loop(
        self,
        rng_key: jax.random.KeyArray,
        model_forward: Callable,
        shape: Tuple,
        conditions: Dict[Tuple[int, int], np.array],
        condition_dim: Optional[int] = None,
        env_ts: Optional[np.array] = None,
        returns_to_go: Optional[np.array] = None,
    ):
        """
        Generate samples from the model.

        :param model_forward: the model apply function without passing params.
        :param shape: the shape of the samples, (N, C, H, W).
        :param returns: if not None, a 1-D array of conditioned returns
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a non-differentiable batch of samples.
        """

        rng_key, sample_key = jax.random.split(rng_key)
        x = self.sample_temperature * jax.random.normal(sample_key, shape)

        timesteps = self.noise_scheduler_state.timesteps
        for t in timesteps:
            x = apply_conditioning(x, conditions, condition_dim)
            ts = np.ones((x.shape[0],), dtype=np.int32) * t

            model_kwargs = {}
            if self.env_ts_condition:
                assert env_ts is not None
                model_kwargs["env_ts"] = env_ts
            if self.returns_condition:
                assert returns_to_go is not None
                model_kwargs["returns_to_go"] = returns_to_go
                model_output_cond = model_forward(
                    None,
                    x,
                    self._scale_timesteps(ts),
                    use_dropout=False,
                    **model_kwargs,
                )
                rng_key, sample_key = jax.random.split(rng_key)
                model_output_uncond = model_forward(
                    sample_key,
                    x,
                    self._scale_timesteps(ts),
                    force_dropout=True,
                    **model_kwargs,
                )
                model_output = model_output_uncond + self.condition_guidance_w * (
                    model_output_cond - model_output_uncond
                )
            else:
                model_output = model_forward(
                    None, x, self._scale_timesteps(ts), **model_kwargs
                )

            rng_key, sample_key = jax.random.split(rng_key)
            if self.use_ddim:
                x = self.noise_scheduler.step(
                    self.noise_scheduler_state, model_output, t, x
                ).prev_sample
            else:
                x = self.noise_scheduler.step(
                    self.noise_scheduler_state, model_output, t, x, sample_key
                ).prev_sample

        x = apply_conditioning(x, conditions, condition_dim)
        return x

    def p_sample_loop_jit(
        self,
        rng_key: jax.random.KeyArray,
        model_forward: Callable,
        shape: Tuple,
        conditions: Dict[Tuple[int, int], np.array],
        condition_dim: Optional[int] = None,
        env_ts: Optional[np.array] = None,
        returns_to_go: Optional[np.array] = None,
    ):
        """
        A loop-jitted version of p_sample_loop().
        It is used for U-Net sampling since unrolling all the loops when using p_sample_loop() is slow.
        It can NOT be used for dql, since currently dql's model_forward is wrapped with `partial`,
        which can not be combined with `flax.linen.while_loop`.
        """

        rng_key, sample_key = jax.random.split(rng_key)
        x = self.sample_temperature * jax.random.normal(sample_key, shape)

        timesteps = self.noise_scheduler_state.timesteps

        def body_fn(mdl, val):
            i, rng_key, x = val
            x = apply_conditioning(x, conditions, condition_dim)
            ts = np.ones((x.shape[0],), dtype=np.int32) * timesteps[i]

            model_kwargs = {}
            if self.env_ts_condition:
                assert env_ts is not None
                model_kwargs["env_ts"] = env_ts
            if self.returns_condition:
                assert returns_to_go is not None
                model_kwargs["returns_to_go"] = returns_to_go
                model_output_cond = mdl(
                    None,
                    x,
                    self._scale_timesteps(ts),
                    use_dropout=False,
                    **model_kwargs,
                )
                rng_key, sample_key = jax.random.split(rng_key)
                model_output_uncond = mdl(
                    sample_key,
                    x,
                    self._scale_timesteps(ts),
                    force_dropout=True,
                    **model_kwargs,
                )
                model_output = model_output_uncond + self.condition_guidance_w * (
                    model_output_cond - model_output_uncond
                )
            else:
                model_output = mdl(None, x, self._scale_timesteps(ts), **model_kwargs)

            rng_key, sample_key = jax.random.split(rng_key)
            if self.use_ddim:
                x = self.noise_scheduler.step(
                    self.noise_scheduler_state, model_output, timesteps[i], x
                ).prev_sample
            else:
                x = self.noise_scheduler.step(
                    self.noise_scheduler_state,
                    model_output,
                    timesteps[i],
                    x,
                    sample_key,
                ).prev_sample
            return i + 1, rng_key, x

        def loop_stop_fn(mdl, c):
            i, _, _ = c
            return i < self.num_inference_steps

        _, _, x = flax.linen.while_loop(
            loop_stop_fn, body_fn, model_forward, (0, rng_key, x)
        )

        return x

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def training_losses(
        self,
        rng_key,
        model_forward,
        x_start,
        conditions,
        t,
        masks=None,
        env_ts=None,
        condition_dim=None,
        returns_to_go=None,
    ):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        rng_key, sample_key = jax.random.split(rng_key)
        noise = jax.random.normal(sample_key, x_start.shape, dtype=x_start.dtype)
        x_t = self.noise_scheduler.add_noise(
            self.noise_scheduler_state, x_start, noise, t
        )
        x_t = apply_conditioning(x_t, conditions, condition_dim)

        model_kwargs = {}
        if env_ts is not None:
            model_kwargs["env_ts"] = env_ts
        if self.returns_condition:
            model_kwargs["returns_to_go"] = returns_to_go

        rng_key, sample_key = jax.random.split(rng_key)
        model_output = model_forward(
            sample_key,
            x_t,
            self._scale_timesteps(t),
            **model_kwargs,
        )

        terms = {"model_output": model_output, "x_t": x_t}
        terms["ts_weights"] = _extract_into_tensor(
            self.normalized_ts_weights, t, x_start.shape[:-1]
        )

        if self.loss_type in (LossType.MSE, LossType.RESCALED_MSE):
            target = noise
            assert model_output.shape == target.shape == x_start.shape

            mse = (target - model_output) ** 2
            if self.loss_weights is not None:
                mse = self.loss_weights * mse
            if masks is not None:
                terms["mse"] = mean_flat(masks * mse) / mean_flat(masks)
            else:
                terms["mse"] = mean_flat(mse)

            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.
    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, np.ndarray):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, np.ndarray) else np.array(x) for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + np.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * np.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.
    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = np.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = np.log(cdf_plus.clip(a_min=1e-12))
    log_one_minus_cdf_min = np.log((1.0 - cdf_min).clip(a_min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = np.where(
        x < -0.999,
        log_cdf_plus,
        np.where(x > 0.999, log_one_minus_cdf_min, np.log(cdf_delta.clip(a_min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr[timesteps].astype(np.float32)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return np.broadcast_to(res, broadcast_shape)
