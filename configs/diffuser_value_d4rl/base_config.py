from ml_collections import ConfigDict, config_dict

from utilities.utils import WandBLogger


def get_base_config():
    config = ConfigDict()
    config.exp_name = config_dict.required_placeholder(str)
    config.log_dir_format = config_dict.required_placeholder(str)

    config.trainer = "ValueFunctionTrainer"
    config.type = "model-free"

    config.env = config_dict.required_placeholder(str)
    config.dataset = "d4rl"
    config.dataset_class = "ValueFunctionDataset"
    config.normalizer = "LimitsNormalizer"
    config.max_traj_length = 1000
    config.horizon = config_dict.required_placeholder(int)
    config.history_horizon = 0
    config.returns_condition = True
    config.env_ts_condition = True
    config.termination_penalty = 0.0
    config.use_padding = True
    config.padding_type = "zero"

    config.seed = 100
    config.batch_size = 32
    config.discount = 1.0
    config.clip_action = 0.999
    config.dim = 32
    config.dim_mults = "1-2-4-8"
    config.kernel_size = 5

    config.n_epochs = config_dict.required_placeholder(int)
    config.n_train_step_per_epoch = config_dict.required_placeholder(int)

    config.evaluator_class = "ValueFunctionOfflineEvaluator"
    config.eval_period = 100
    config.num_eval_envs = 10
    config.eval_env_seed = 0
    config.eval_batch_size = 128

    config.save_period = 0
    config.logging = WandBLogger.get_default_config()

    config.algo_cfg = ConfigDict()
    config.algo_cfg.horizon = config.get_ref("horizon")
    config.algo_cfg.history_horizon = config.get_ref("history_horizon")
    config.algo_cfg.sample_temperature = 0.5
    config.algo_cfg.num_timesteps = 200
    config.algo_cfg.schedule_name = "cosine"
    # learning related
    config.algo_cfg.lr = 2e-4
    config.algo_cfg.lr_decay = False
    config.algo_cfg.lr_decay_steps = 1000000
    config.algo_cfg.max_grad_norm = 0.0
    config.algo_cfg.weight_decay = 0.0
    # for ema decay
    config.algo_cfg.ema_decay = 0.995
    config.algo_cfg.step_start_ema = 2000
    config.algo_cfg.update_ema_every = 10

    return config
