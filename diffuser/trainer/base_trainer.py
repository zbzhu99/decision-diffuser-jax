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

import importlib

import absl
import absl.flags
import gym
import jax
import jax.numpy as jnp
import torch
import tqdm

from diffuser.constants import DATASET, DATASET_MAP, ENV_MAP
from diffuser.hps import hyperparameters
from utilities.data_utils import cycle, numpy_collate
from utilities.jax_utils import batch_to_jax
from utilities.sampler import TrajSampler
from utilities.utils import (
    DotFormatter,
    Timer,
    WandBLogger,
    get_user_flags,
    prefix_metrics,
)
from viskit.logging import logger, setup_logger


class BaseTrainer:
    def __init__(self, config, use_absl: bool = True):
        if use_absl:
            self._cfgs = absl.flags.FLAGS
        else:
            self._cfgs = config

        self._cfgs.algo_cfg.max_grad_norm = hyperparameters[self._cfgs.env]["gn"]
        self._cfgs.algo_cfg.lr_decay_steps = (
            self._cfgs.n_epochs * self._cfgs.n_train_step_per_epoch
        )

        if self._cfgs.activation == "mish":
            act_fn = lambda x: x * jnp.tanh(jax.nn.softplus(x))
        else:
            act_fn = getattr(jax.nn, self._cfgs.activation)

        self._act_fn = act_fn
        self._variant = get_user_flags(self._cfgs, config)

        # get high level env
        env_name_full = self._cfgs.env
        for scenario_name in ENV_MAP:
            if scenario_name in env_name_full:
                self._env = ENV_MAP[scenario_name]
                break
        else:
            raise NotImplementedError

    def train(self):
        self._setup()

        viskit_metrics = {}
        for epoch in range(self._cfgs.n_epochs):
            metrics = {"epoch": epoch}

            with Timer() as eval_timer:
                if self._cfgs.eval_period > 0 and epoch % self._cfgs.eval_period == 0:
                    self._evaluator.update_params(self._agent.eval_params)
                    eval_metrics = self._evaluator.evaluate(epoch)
                    metrics.update(eval_metrics)

                if self._cfgs.save_period > 0 and epoch % self._cfgs.save_period == 0:
                    self._save_model(epoch)

            with Timer() as train_timer:
                for _ in tqdm.tqdm(range(self._cfgs.n_train_step_per_epoch)):
                    batch = batch_to_jax(next(self._dataloader))
                    metrics.update(prefix_metrics(self._agent.train(batch), "agent"))

            metrics["train_time"] = train_timer()
            metrics["eval_time"] = eval_timer()
            metrics["epoch_time"] = train_timer() + eval_timer()
            self._wandb_logger.log(metrics)
            viskit_metrics.update(metrics)
            logger.record_dict(viskit_metrics)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)

        # save model at final epoch
        if (
            self._cfgs.save_period > 0
            and self._cfgs.n_epochs % self._cfgs.save_period == 0
        ):
            self._save_model(self._cfgs.n_epochs)

        if (
            self._cfgs.eval_period > 0
            and self._cfgs.n_epochs % self._cfgs.eval_period == 0
        ):
            self._evaluator.update_params(self._agent.eval_params)
            self._evaluator.evaluate(self._cfgs.n_epochs)

    def _setup(self):
        raise NotImplementedError

    def _save_model(self, epoch: int):
        save_data = {
            "agent_states": self._agent.train_states,
            "variant": self._variant,
            "epoch": epoch,
        }
        logger.save_orbax_checkpoint(save_data, f"checkpoints/model_{epoch}")

    def _setup_logger(self):
        logging_configs = self._cfgs.logging
        logging_configs["log_dir"] = DotFormatter().vformat(
            self._cfgs.log_dir_format, [], self._variant
        )
        wandb_logger = WandBLogger(config=logging_configs, variant=self._variant)
        setup_logger(
            variant=self._variant,
            log_dir=wandb_logger.output_dir,
            seed=self._cfgs.seed,
            include_exp_prefix_sub_dir=False,
        )
        return wandb_logger

    def _setup_d4rl(self):
        from data.d4rl import get_dataset

        if self._cfgs.dataset_class in ["QLearningDataset"]:
            include_next_obs = True
        else:
            include_next_obs = False

        eval_sampler = TrajSampler(
            lambda: gym.make(self._cfgs.env),
            self._cfgs.num_eval_envs,
            self._cfgs.eval_env_seed,
            self._cfgs.max_traj_length,
            use_env_ts=self._cfgs.env_ts_condition,
            history_horizon=getattr(self._cfgs, "history_horizon", 0),
            update_rtg=getattr(self._cfgs, "update_rtg", False),
            padding_type=self._cfgs.padding_type,
        )
        dataset = get_dataset(
            eval_sampler.env,
            discount=self._cfgs.discount,
            horizon=self._cfgs.horizon,
            max_traj_length=self._cfgs.max_traj_length,
            include_next_obs=include_next_obs,
            termination_penalty=self._cfgs.termination_penalty,
        )
        return dataset, eval_sampler

    def _setup_dataset(self):
        dataset_type = DATASET_MAP[self._cfgs.dataset]
        if dataset_type == DATASET.D4RL:
            dataset, eval_sampler = self._setup_d4rl()
        else:
            raise NotImplementedError

        dataset = getattr(
            importlib.import_module("data.sequence"), self._cfgs.dataset_class
        )(
            dataset,
            horizon=self._cfgs.horizon,
            history_horizon=getattr(self._cfgs, "history_horizon", 0),
            max_traj_length=self._cfgs.max_traj_length,
            include_returns=self._cfgs.returns_condition,
            include_env_ts=self._cfgs.env_ts_condition,
            normalizer=self._cfgs.normalizer,
            use_inv_dynamic=getattr(self._cfgs, "use_inv_dynamic", True),
            use_padding=self._cfgs.use_padding,
            padding_type=self._cfgs.padding_type,
        )
        eval_sampler.set_normalizer(dataset.normalizer)

        self._observation_dim = eval_sampler.env.observation_space.shape[0]
        self._action_dim = eval_sampler.env.action_space.shape[0]

        return dataset, eval_sampler

    def _setup_evaluator(self, sampler_policy, eval_sampler, dataset):
        evaluator_class = getattr(
            importlib.import_module("diffuser.evaluator"), self._cfgs.evaluator_class
        )

        if evaluator_class.eval_mode == "online":
            evaluator = evaluator_class(self._cfgs, sampler_policy, eval_sampler)
        elif evaluator_class.eval_mode == "offline":
            eval_data_sampler = torch.utils.data.RandomSampler(dataset)
            eval_dataloader = cycle(
                torch.utils.data.DataLoader(
                    dataset,
                    sampler=eval_data_sampler,
                    batch_size=self._cfgs.eval_batch_size,
                    collate_fn=numpy_collate,
                    drop_last=True,
                    num_workers=4,
                )
            )
            evaluator = evaluator_class(self._cfgs, sampler_policy, eval_dataloader)
        elif evaluator_class.eval_mode == "skip":
            evaluator = evaluator_class(self._cfgs, sampler_policy)
        else:
            raise NotImplementedError(f"Unknown eval_mode: {self._cfgs.eval_mode}")

        return evaluator
