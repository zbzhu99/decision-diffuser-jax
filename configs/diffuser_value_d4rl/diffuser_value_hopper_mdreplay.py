from configs.diffuser_value_d4rl.base_config import get_base_config


def get_config():
    config = get_base_config()
    config.exp_name = "diffuser_value_d4rl"
    config.log_dir_format = (
        "{exp_name}/{env}/h_{horizon}-hh_{history_horizon}-dis_{discount}/{seed}"
    )

    config.env = "hopper-medium-replay-v2"
    config.discount = 0.99
    config.termination_penalty = -100.0
    config.normalizer = "LimitsNormalizer"

    config.horizon = 100
    config.history_horizon = 20
    config.padding_type = "same"

    config.n_epochs = 1000
    config.n_train_step_per_epoch = 1000

    config.save_period = 100

    config.algo_cfg.num_timesteps = 20

    return config
