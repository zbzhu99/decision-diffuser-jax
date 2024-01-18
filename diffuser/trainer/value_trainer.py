import torch
import importlib

from diffuser.algos import DiffuserValue
from diffuser.diffusion import ValueDiffusion, LossType
from diffuser.nets import ValueFunction
from diffuser.trainer.base_trainer import BaseTrainer
from utilities.data_utils import cycle, numpy_collate
from utilities.utils import set_random_seed, to_arch


class ValueFunctionTrainer(BaseTrainer):
    def _setup(self):
        set_random_seed(self._cfgs.seed)
        # setup logger
        self._wandb_logger = self._setup_logger()

        # setup dataset and eval_sample
        dataset, eval_sampler = self._setup_dataset()
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

        # setup value function
        self._value_function = self._setup_value_function()

        # setup agent
        self._agent = DiffuserValue(self._cfgs.algo_cfg, self._value_function)

        # setup evaluator
        self._evaluator = self._setup_evaluator(self._agent, dataset)

    def _setup_value_function(self):
        gd = ValueDiffusion(
            num_timesteps=self._cfgs.algo_cfg.num_timesteps,
            schedule_name=self._cfgs.algo_cfg.schedule_name,
            loss_type=LossType.MSE,
        )
        value_function = ValueFunction(
            diffusion=gd,
            sample_dim=self._observation_dim + self._action_dim,
            action_dim=self._action_dim,
            dim=self._cfgs.dim,
            dim_mults=to_arch(self._cfgs.dim_mults),
            kernel_size=self._cfgs.kernel_size,
        )
        return value_function

    def _setup_evaluator(self, agent, dataset):
        evaluator_class = getattr(
            importlib.import_module("diffuser.evaluator"), self._cfgs.evaluator_class
        )
        assert (
            evaluator_class.eval_mode == "offline"
        ), "Evaluator must be offline for value training"

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
        evaluator = evaluator_class(self._cfgs, agent, eval_dataloader)
        return evaluator
