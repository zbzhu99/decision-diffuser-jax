import torch

from diffuser.algos import DecisionDiffuser
from diffuser.diffusion import GaussianDiffusion, LossType, ModelMeanType, ModelVarType
from diffuser.nets import DiffusionPlanner, InverseDynamic
from diffuser.policy import DiffuserPolicy
from diffuser.trainer.base_trainer import BaseTrainer
from utilities.data_utils import cycle, numpy_collate
from utilities.utils import set_random_seed, to_arch


class DiffuserTrainer(BaseTrainer):
    def _setup(self):
        set_random_seed(self._cfgs.seed)
        # setup logger
        self._wandb_logger = self._setup_logger()

        # setup dataset and eval_sample
        dataset, eval_sampler = self._setup_dataset()
        eval_sampler.set_target_return(self._cfgs.target_return, self._cfgs.discount)
        data_sampler = torch.utils.data.RandomSampler(dataset)
        self._dataloader = cycle(
            torch.utils.data.DataLoader(
                dataset,
                sampler=data_sampler,
                batch_size=self._cfgs.batch_size,
                collate_fn=numpy_collate,
                drop_last=True,
                num_workers=8,
            )
        )

        # setup policy
        self._planner, self._inv_model = self._setup_policy()

        # setup agent
        self._agent = DecisionDiffuser(
            self._cfgs.algo_cfg, self._planner, self._inv_model
        )

        # setup evaluator
        sampler_policy = DiffuserPolicy(self._planner, self._inv_model)
        self._evaluator = self._setup_evaluator(sampler_policy, eval_sampler, dataset)

    def _setup_policy(self):
        gd = GaussianDiffusion(
            num_timesteps=self._cfgs.algo_cfg.num_timesteps,
            schedule_name=self._cfgs.algo_cfg.schedule_name,
            model_mean_type=ModelMeanType.EPSILON,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            returns_condition=self._cfgs.returns_condition,
            condition_guidance_w=self._cfgs.condition_guidance_w,
            sample_temperature=self._cfgs.algo_cfg.sample_temperature,
        )

        if self._cfgs.use_inv_dynamic:
            inv_model = InverseDynamic(
                action_dim=self._action_dim,
                hidden_dims=to_arch(self._cfgs.inv_hidden_dims),
            )
            plan_sample_dim = self._observation_dim
            plan_action_dim = 0
        else:
            inv_model = None
            plan_sample_dim = self._observation_dim + self._action_dim
            plan_action_dim = self._action_dim

        planner = DiffusionPlanner(
            diffusion=gd,
            horizon=self._cfgs.horizon,
            history_horizon=self._cfgs.history_horizon,
            sample_dim=plan_sample_dim,
            action_dim=plan_action_dim,
            dim=self._cfgs.dim,
            dim_mults=to_arch(self._cfgs.dim_mults),
            env_ts_condition=self._cfgs.env_ts_condition,
            returns_condition=self._cfgs.returns_condition,
            condition_dropout=self._cfgs.condition_dropout,
            kernel_size=self._cfgs.kernel_size,
            sample_method=self._cfgs.sample_method,
            dpm_steps=self._cfgs.algo_cfg.dpm_steps,
            dpm_t_end=self._cfgs.algo_cfg.dpm_t_end,
            max_traj_length=self._cfgs.max_traj_length,
        )
        return planner, inv_model
