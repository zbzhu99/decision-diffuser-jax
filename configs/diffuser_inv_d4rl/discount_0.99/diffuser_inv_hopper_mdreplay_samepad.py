from configs.diffuser_inv_d4rl.base_config import get_base_config


def get_config():
    config = get_base_config()
    config.exp_name = "diffuser_inv_d4rl_samepad_cdfnorm"
    config.log_dir_format = "{exp_name}/{env}/h_{horizon}-hh_{history_horizon}-tr_{target_return}-dis_{discount}-update-rtg_{update_rtg}-envts_{env_ts_condition}/{seed}"

    config.env = "hopper-medium-replay-v2"
    config.update_rtg = False
    config.env_ts_condition = False
    config.target_return = 320.0
    config.discount = 0.99
    config.termination_penalty = -100.0
    config.normalizer = "CDFNormalizer"

    config.horizon = 20
    config.history_horizon = 4
    config.padding_type = "same"

    config.n_epochs = 1000
    config.n_train_step_per_epoch = 1000

    config.save_period = 0

    config.algo_cfg.use_ddim = True
    config.algo_cfg.n_ddim_steps = 15

    return config
