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

from typing import List

import numpy as np
import torch

from utilities.normalization import DatasetNormalizer


class SequenceDataset(torch.utils.data.Dataset):
    """DataLoader with customized sampler."""

    def __init__(
        self,
        data: dict,
        horizon: int,
        max_traj_length: int,
        normalizer: str = "LimitsNormalizer",
        discrete_action: bool = False,
        use_action: bool = True,
        use_inv_dynamic: bool = True,
        include_returns: bool = True,
        include_env_ts: bool = True,
    ) -> None:
        self.include_returns = include_returns
        self.include_env_ts = include_env_ts
        self.use_action = use_action
        self.use_inverse_dynamic = use_inv_dynamic
        self.max_traj_length = max_traj_length
        self.horizon = horizon
        self.discrete_action = discrete_action
        if discrete_action:
            raise NotImplementedError

        self._data = data
        self.normalizer = DatasetNormalizer(self._data, normalizer)

        self._keys = list(data.keys()).remove("traj_lengths")
        self._indices = self.make_indices()

        self.n_episodes = len(self._data)
        self.normalize()
        print(self._data)

    def __len__(self):
        return len(self._indices)

    def make_indices(self):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """

        indices = []
        for i, traj_length in enumerate(self._data["traj_lengths"]):
            # get `end` and `mask_end` for each `start`
            for start in range(traj_length - 1):
                end = start + self.horizon
                mask_end = min(end, traj_length)
                indices.append((i, start, end, mask_end))
        indices = np.array(indices)
        return indices

    def normalize(self, keys: List[str] = None) -> None:
        """
        normalize fields that will be predicted by the diffusion model
        """
        if keys is None:
            keys = ["observations", "returns"]
            if self.use_action and not self.discrete_action:
                keys.append("actions")

        for key in keys:
            shape = self._data[key].shape
            array = self._data[key].reshape(shape[0] * shape[1], *shape[2:])
            normed = self.normalizer(array, key)
            self._data[f"normed_{key}"] = normed.reshape(shape)

    def get_conditions(self, observations):
        """
        condition on current observation for planning
        """

        return {0: observations[0]}

    def __getitem__(self, idx):
        path_ind, start, end, mask_end = self._indices[idx]

        observations = self._data.normed_observations[path_ind, start:end]
        if self.use_action:
            if self.discrete_action:
                actions = self._data.actions[path_ind, start:end]
            else:
                actions = self._data.normed_actions[path_ind, start:end]

        masks = np.zeros((observations.shape[0], 1))
        if self.use_inverse_dynamic:
            masks[1: mask_end - start] = 1.0
        else:
            masks[: mask_end - start] = 1.0

        conditions = self.get_conditions(observations)
        ret_dict = dict(samples=observations, conditions=conditions, masks=masks)

        if self.include_env_ts:
            ret_dict["env_ts"] = start
        if self.include_returns:
            ret_dict["returns"] = self._data.normed_returns[path_ind, start].reshape(
                1, 1
            )

        if self.use_action:
            ret_dict["actions"] = actions

        return ret_dict


class QLearningDataset(SequenceDataset):
    def make_indices(self):
        """
        makes indices for sampling from dataset;
        each index maps to a datapoint
        """

        assert self.horizon == 1, "QLearningDataset only supports horizon=1"
        indices = []
        for i, traj_length in enumerate(self._data["traj_lengths"]):
            # get `end` and `mask_end` for each `start`
            for start in range(traj_length):
                end = start + self.horizon
                mask_end = min(end, traj_length)
                indices.append((i, start, end, mask_end))
        indices = np.array(indices)
        return indices

    def normalize(self, keys: List[str] = None) -> None:
        """
        normalize fields that will be predicted by the diffusion model
        """

        super().normalize(keys)

        shape = self._data["next_observations"].shape
        array = self._data["next_observations"].reshape(
            shape[0] * shape[1], *shape[2:]
        )
        normed = self.normalizer(array, "observations")
        self._data["normed_next_observations"] = normed.reshape(shape)

    def get_conditions(self, observations):
        return {}

    def __getitem__(self, idx):
        path_ind, start, end, mask_end = self._indices[idx]

        observations = self._data.normed_observations[path_ind, start:end].squeeze(0)
        actions = self._data.normed_actions[path_ind, start:end].squeeze(0)
        rewards = self._data.rewards[path_ind, start:end].squeeze(0)
        next_observations = self._data.normed_next_observations[
            path_ind, start:end
        ].squeeze(0)
        dones = self._data.terminals[path_ind, start:end].squeeze(0)

        conditions = self.get_conditions(observations)
        next_conditions = self.get_conditions(next_observations)

        ret_dict = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            conditions=conditions,
            next_conditions=next_conditions,
        )
        return ret_dict
