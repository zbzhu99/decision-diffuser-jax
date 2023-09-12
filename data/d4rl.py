from functools import partial
from typing import Callable

import d4rl  # noqa
import gym

from utilities.utils import compose

from .dataset import Dataset
from .preprocess import (
    add_discounted_returns,
    clip_actions,
    pad_trajs_to_dataset,
    split_to_trajs,
)


def get_dataset(
    env,
    max_traj_length: int,
    horizon: int,
    discount: float = 0.99,
    termination_penalty: float = None,
    include_next_obs: bool = False,
    clip_to_eps: bool = False,  # disable action clip for debugging purpose
):
    preprocess_fn = compose(
        partial(
            pad_trajs_to_dataset,
            horizon=horizon,
            max_traj_length=max_traj_length,
            include_next_obs=include_next_obs,
        ),
        partial(
            add_discounted_returns,
            discount=discount,
            termination_penalty=termination_penalty,
        ),
        split_to_trajs,
        partial(
            clip_actions,
            clip_to_eps=clip_to_eps,
        ),
    )
    return D4RLDataset(env, preprocess_fn=preprocess_fn)


class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, preprocess_fn: Callable, **kwargs):
        self.raw_dataset = dataset = env.get_dataset()
        data_dict = preprocess_fn(dataset)
        super().__init__(**data_dict, **kwargs)
