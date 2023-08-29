import numpy as np

from utilities.data_utils import atleast_nd


def clip_actions(dataset, clip_to_eps: bool = True, eps: float = 1e-5):
    if clip_to_eps:
        lim = 1 - eps
        dataset["actions"] = np.clip(dataset["actions"], -lim, lim)
    return dataset


def compute_discounted_cumsum_returns(traj, gamma: float) -> np.ndarray:
    """
    Calculate the discounted cumulative reward sum of traj
    """

    cumsum = np.zeros(len(traj))
    cumsum[-1] = traj[-1][2]
    for t in reversed(range(cumsum.shape[0] - 1)):
        cumsum[t] = traj[t][2] + gamma * cumsum[t + 1]
    return cumsum


def add_discounted_returns(
    trajs,
    discount: float,
    termination_penalty: float,
):
    for traj in trajs:
        if np.any([bool(step[4]) for step in traj]) and termination_penalty is not None:
            traj[-1][2] += termination_penalty
        reward_returns = compute_discounted_cumsum_returns(traj, discount)
        for idx, step in enumerate(traj):
            step.append(reward_returns[idx])
    return trajs


def split_to_trajs(dataset):
    dones_float = np.zeros_like(dataset["rewards"])  # truncated and terminal
    for i in range(len(dones_float) - 1):
        if (
            np.linalg.norm(
                dataset["observations"][i + 1] - dataset["next_observations"][i]
            )
            > 1e-6
            or dataset["terminals"][i] == 1.0
        ):
            dones_float[i] = 1
        else:
            dones_float[i] = 0
    dones_float[-1] = 1

    trajs = [[]]
    for i in range(len(dataset["observations"])):
        trajs[-1].append(
            [
                dataset["observations"][i],
                dataset["actions"][i],
                dataset["rewards"][i],
                dones_float[i],
                dataset["terminals"][i],
                dataset["next_observations"][i],
            ]
        )
        if dones_float[i] == 1.0 and i + 1 < len(dataset["observations"]):
            trajs.append([])

    return trajs


def pad_trajs_to_dataset(
    trajs,
    max_traj_length: int,
    include_next_obs: bool = False,
):
    n_trajs = len(trajs)

    dataset = {}
    obs_dim, act_dim = trajs[0][0][0].shape[0], trajs[0][0][1].shape[0]
    dataset["observations"] = np.zeros(
        (n_trajs, max_traj_length, obs_dim), dtype=np.float32
    )
    dataset["actions"] = np.zeros((n_trajs, max_traj_length, act_dim), dtype=np.float32)
    dataset["rewards"] = np.zeros((n_trajs, max_traj_length), dtype=np.float32)
    dataset["terminals"] = np.zeros((n_trajs, max_traj_length), dtype=np.float32)
    dataset["dones_float"] = np.zeros((n_trajs, max_traj_length), dtype=np.float32)
    dataset["traj_lengths"] = np.zeros((n_trajs,), dtype=np.int32)
    dataset["returns"] = np.zeros((n_trajs, max_traj_length), dtype=np.float32)
    if include_next_obs:
        dataset["next_observations"] = np.zeros(
            (n_trajs, max_traj_length, obs_dim), dtype=np.float32
        )

    for idx, traj in enumerate(trajs):
        traj_length = len(traj)
        dataset["traj_lengths"][idx] = traj_length
        dataset["observations"][idx, :traj_length] = atleast_nd(
            np.stack([ts[0] for ts in traj], axis=0),
            n=2,
        )
        dataset["actions"][idx, :traj_length] = atleast_nd(
            np.stack([ts[1] for ts in traj], axis=0),
            n=2,
        )
        dataset["rewards"][idx, :traj_length] = np.stack([ts[2] for ts in traj], axis=0)
        dataset["dones_float"][idx, :traj_length] = np.stack(
            [ts[3] for ts in traj], axis=0
        )
        dataset["terminals"][idx, :traj_length] = np.stack(
            [bool(ts[4]) for ts in traj], axis=0
        )
        if include_next_obs:
            dataset["next_observations"][idx, :traj_length] = atleast_nd(
                np.stack([ts[5] for ts in traj], axis=0),
                n=2,
            )
        dataset["returns"][idx, :traj_length] = np.stack(
            [ts[-1] for ts in traj], axis=0
        )

    return dataset
