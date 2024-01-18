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

"""Agent trajectory samplers."""

import time
from collections import deque
from typing import Callable

import numpy as np

from env import get_envs

WIDTH = 250
HEIGHT = 200


class StepSampler(object):
    def __init__(self, env, max_traj_length=1000):
        self.max_traj_length = max_traj_length
        self._env = env
        self._traj_steps = 0
        self._current_observation = self.env.reset()

    def sample(self, policy, n_steps, deterministic=False, replay_buffer=None):
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for _ in range(n_steps):
            self._traj_steps += 1
            observation = self._current_observation
            action = policy(
                observation.reshape(1, -1), deterministic=deterministic
            ).reshape(-1)
            next_observation, reward, done, _ = self.env.step(action)
            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_observations.append(next_observation)

            if replay_buffer is not None:
                replay_buffer.add_sample(
                    observation, action, reward, next_observation, done
                )

            self._current_observation = next_observation

            if done or self._traj_steps >= self.max_traj_length:
                self._traj_steps = 0
                self._current_observation = self.env.reset()

        return dict(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            next_observations=np.array(next_observations, dtype=np.float32),
            dones=np.array(dones, dtype=np.float32),
        )

    @property
    def env(self):
        return self._env


class TrajSampler(object):
    def __init__(
        self,
        env_fn: Callable,
        num_envs: int,
        seed: int,
        max_traj_length: int = 1000,
        render: bool = False,
        use_env_ts: bool = False,
        history_horizon: int = 0,
        update_rtg: bool = False,
        padding_type: str = "zero",
    ):
        self.max_traj_length = max_traj_length
        self.use_env_ts = use_env_ts
        self.history_horizon = history_horizon
        self.padding_type = padding_type
        self.update_rtg = update_rtg
        self._env = env_fn()
        self._envs = get_envs(env_fn, num_envs)
        self._envs.seed(seed)
        self._num_envs = num_envs
        self._render = render
        self._normalizer = None
        self._target_return = None
        self._discount = 1.0

    def set_normalizer(self, normalizer):
        self._normalizer = normalizer

    def set_target_return(self, target_return: float, discount: float = 1.0):
        self._target_return = target_return
        self._discount = discount

    def sample(
        self,
        policy,
        n_trajs: int,
        deterministic: bool = False,
        env_render_fn: str = "render",
    ):
        assert n_trajs > 0
        ready_env_ids = np.arange(min(self._num_envs, n_trajs))
        if self._target_return is not None:
            returns_to_go = np.ones(len(ready_env_ids)) * self._target_return
        env_ts = np.zeros(len(ready_env_ids), dtype=np.int32)

        observation, _ = self.envs.reset(ready_env_ids)
        observation = self._normalizer.normalize(observation, "observations")

        if self.history_horizon > 0:
            obs_queue = deque(maxlen=self.history_horizon + 1)
            if self.padding_type == "zero":
                obs_queue.extend(
                    [np.zeros_like(observation) for _ in range(self.history_horizon)]
                )
            elif self.padding_type == "same":
                obs_queue.extend([observation for _ in range(self.history_horizon)])
            else:
                raise ValueError(f"Invalid padding type {self.padding_type}")

        observations = [[] for i in range(len(ready_env_ids))]
        actions = [[] for _ in range(len(ready_env_ids))]
        rewards = [[] for _ in range(len(ready_env_ids))]
        next_observations = [[] for _ in range(len(ready_env_ids))]
        dones = [[] for _ in range(len(ready_env_ids))]

        trajs = []
        n_finished_trajs = 0
        while True:
            policy_kwargs = {}
            if self._target_return is not None:
                policy_kwargs["returns_to_go"] = self._normalizer.normalize(
                    returns_to_go[ready_env_ids], "returns"
                )
            if self.use_env_ts:
                policy_kwargs["env_ts"] = env_ts[ready_env_ids]

            if self.history_horizon > 0:
                obs_queue.append(observation)
                full_observation = np.stack(list(obs_queue), axis=1)
            else:
                full_observation = np.expand_dims(observation, axis=1)

            action = policy(
                full_observation, deterministic=deterministic, **policy_kwargs
            )
            action = self._normalizer.unnormalize(action, "actions")

            next_observation, reward, terminated, truncated, _ = self.envs.step(
                action, ready_env_ids
            )

            env_ts[ready_env_ids] += 1
            # ret0 = r0 + r1 * discount + r2 * discount^2 + ...
            # ret1 = r1 + r2 * discount + r3 * discount^2 + ... = (ret0 - r0) / discount
            if self._target_return is not None and self.update_rtg:
                returns_to_go[ready_env_ids] = (
                    returns_to_go[ready_env_ids] - reward
                ) / self._discount

            done = np.logical_or(terminated, truncated)
            if self._render:
                getattr(self.envs, env_render_fn)()
                time.sleep(0.01)

            next_observation = self._normalizer.normalize(
                next_observation, "observations"
            )

            for idx, env_id in enumerate(ready_env_ids):
                observations[env_id].append(observation[idx])
                actions[env_id].append(action[idx])
                rewards[env_id].append(reward[idx])
                next_observations[env_id].append(next_observation[idx])
                dones[env_id].append(done[idx])

            if np.any(done):
                env_ind_local = np.where(done)[0]
                env_ind_global = ready_env_ids[env_ind_local]

                for ind in env_ind_local:
                    trajs.append(
                        dict(
                            observations=np.array(observations[ind], dtype=np.float32),
                            actions=np.array(actions[ind], dtype=np.float32),
                            rewards=np.array(rewards[ind], dtype=np.float32),
                            next_observations=np.array(
                                next_observations[ind], dtype=np.float32
                            ),
                            dones=np.array(dones[ind], dtype=np.float32),
                        )
                    )
                    observations[ind] = []
                    actions[ind] = []
                    rewards[ind] = []
                    next_observations[ind] = []
                    dones[ind] = []

                if self._target_return is not None and self.update_rtg:
                    returns_to_go[env_ind_global] = self._target_return
                env_ts[env_ind_global] = 0

                n_finished_trajs += len(env_ind_local)
                if n_finished_trajs >= n_trajs:
                    trajs = trajs[:n_trajs]
                    break

                # surplus_env_num = len(ready_env_ids) - (n_trajs - n_finished_trajs)
                # if surplus_env_num > 0:
                #     mask = np.ones_like(ready_env_ids, dtype=bool)
                #     mask[env_ind_local[:surplus_env_num]] = False
                #     ready_env_ids = ready_env_ids[mask]

                obs_reset, _ = self.envs.reset(env_ind_global)
                obs_reset = self._normalizer.normalize(obs_reset, "observations")
                next_observation[env_ind_global] = obs_reset

                if self.history_horizon > 0:
                    for i in range(len(obs_queue)):
                        if self.padding_type == "zero":
                            obs_queue[i][env_ind_global] = 0.0
                        elif self.padding_type == "same":
                            obs_queue[i][env_ind_global] = obs_reset
                        else:
                            raise ValueError(
                                f"Invalid padding type {self.padding_type}"
                            )

            observation = next_observation
        return trajs

    @property
    def env(self):
        return self._env

    @property
    def envs(self):
        return self._envs
