#!/bin/env/python3

"""
This script includes a wrapper class for the pettingzoo MA-environment class,
which augments each agent's observation with a random gaussian vector, which
is sampled at the beginning of each episode
"""
import numpy as np
from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from gym.spaces import Box, Dict


class FlattenEnvWrapper(BaseWrapper):
    def __init__(self, original_env: SimpleEnv) -> None:
        super(FlattenEnvWrapper, self).__init__(original_env)
        self._env = original_env
        self._update_observation_space()

    def _update_observation_space(self) -> None:
        """
        Update the observation space with augmented dimensions
        """
        for agent, space in self.observation_spaces.items():
            obs_space = space["observation"]
            assert len(obs_space.shape) > 1
            flatten_dim = np.prod(obs_space.shape)
            dtype = obs_space.dtype if not 'int' in str(obs_space.dtype) else np.float32
            new_space = Box(low=np.full(flatten_dim, -np.inf),
                                    high=np.full(flatten_dim, np.inf),
                                    dtype=dtype)
            self.observation_spaces[agent] = new_space

    def reset(self) -> None:
        super(FlattenEnvWrapper, self).reset()

    def observe(self, agent) -> np.ndarray:
        original_obs = self.env.observe(agent)["observation"]
        dtype = original_obs.dtype if not 'int' in str(original_obs.dtype) else np.float32
        new_obs = original_obs.flatten().astype(dtype)
        return new_obs

