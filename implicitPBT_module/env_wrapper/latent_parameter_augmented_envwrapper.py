#!/bin/env/python3

"""
This script includes a wrapper class for the pettingzoo MA-environment class,
which augments each agent's observation with a random gaussian vector, which
is sampled at the beginning of each episode
"""
import numpy as np
from pettingzoo.utils.wrappers import BaseWrapper
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Box, Dict


class LatentGaussianAugmentedPettiingZooEnvWrapper(BaseWrapper):
    def __init__(self, original_env: SimpleEnv, 
                latent_parameter_dim: int,
                gaussian_std=1,
                use_dict_obs_space: bool=True) -> None:
        super(LatentGaussianAugmentedPettiingZooEnvWrapper, self).__init__(original_env)
        self._env = original_env
        self._latent_parameter_dim = latent_parameter_dim
        self._latent_parameter = None
        self._gaussian_std = gaussian_std
        self._use_dict_obs_space = use_dict_obs_space
        self.reset()
        self.agents = self._env.possible_agents
        self._index_map = {agent: idx for idx, agent in enumerate(self.agents)}
        self._augment_observation_space()

    def _augment_observation_space(self) -> None:
        """
        Update the observation space with augmented dimensions
        """
        for agent, space in self.observation_spaces.items():
            if self._use_dict_obs_space:
                spaces = {
                    'original_obs': space,
                    'random_noise': Box(low=np.full(self._latent_parameter_dim, -np.inf),
                                        high=np.full(self._latent_parameter_dim, np.inf),
                                        dtype=np.float32)
                }
                # This 'Dict' is gym.spaces.Dict
                new_space = Dict(spaces)
            else:
                if not isinstance(space, Box):
                    raise TypeError("Cannot handle spaces that are not box now")
                if len(space.shape) > 1:
                    raise TypeError("Cannot handle non-flat spaces now")
                original_low = space.low
                original_high = space.high
                low = np.concatenate([original_low, np.full(self._latent_parameter_dim, -np.inf)])
                high = np.concatenate([original_high, np.full(self._latent_parameter_dim, np.inf)])
                new_space = Box(low=low, high=high, dtype=np.float32)

            self.observation_spaces[agent] = new_space

    def _sample_latent_parameter(self) -> None:
        self._latent_parameter = self._gaussian_std * np.random.randn(len(self._env.possible_agents),
                                                                    self._latent_parameter_dim)
        if str(self._env) == 'sc2':
            assert self.num_agents == 6
            # for starcraft env, sync same team member latent param
            self._latent_parameter[1, :] = self._latent_parameter[0, :]
            self._latent_parameter[2, :] = self._latent_parameter[0, :]
            self._latent_parameter[4, :] = self._latent_parameter[3, :]
            self._latent_parameter[5, :] = self._latent_parameter[3, :]

    def reset(self) -> None:
        super(LatentGaussianAugmentedPettiingZooEnvWrapper, self).reset()
        self._sample_latent_parameter()

    def observe(self, agent) -> np.ndarray:
        original_obs = self.env.observe(agent)
        latent_parameter_obs = self._latent_parameter[self._index_map[agent], :]
        if self._use_dict_obs_space:
            joint_obs = {'original_obs': original_obs, 'random_noise': latent_parameter_obs}
            return joint_obs
        joint_obs = np.concatenate([original_obs, latent_parameter_obs])
        return joint_obs


class LatentGaussianAugmentedMultiAgentEnvWrapper(MultiAgentEnv):
    def __init__(self, original_env: MultiAgentEnv,
                 latent_parameter_dim: int,
                 gaussian_std=1,
                 use_dict_obs_space: bool=True) -> None:
        self._env = original_env
        self._latent_parameter_dim = latent_parameter_dim
        self._latent_parameter = None
        self._gaussian_std = gaussian_std
        self._use_dict_obs_space = use_dict_obs_space
        self._sample_latent_parameter()
        self.agents = self._env.agents
        self._index_map = {agent: idx for idx, agent in enumerate(self.agents)}
        self._augment_observation_space()
        self.action_spaces = self._env.action_spaces
        self.observation_space = self.observation_spaces[self.agents[0]]
        self.action_space = self.action_spaces[self.agents[0]]

    def _augment_observation_space(self) -> None:
        """
        Update the observation space with augmented dimensions
        """
        self.observation_spaces = {agent: None for agent in self.agents}
        for agent, space in self._env.observation_spaces.items():
            if self._use_dict_obs_space:
                spaces = {
                    'original_obs': space,
                    'random_noise': Box(low=np.full(self._latent_parameter_dim, -np.inf),
                                        high=np.full(self._latent_parameter_dim, np.inf),
                                        dtype=np.float32)
                }
                # This 'Dict' is gym.spaces.Dict
                new_space = Dict(spaces)
            else:
                if not isinstance(space, Box):
                    raise TypeError("Cannot handle spaces that are not box now")
                if len(space.shape) > 1:
                    raise TypeError("Cannot handle non-flat spaces now")
                original_low = space.low
                original_high = space.high
                low = np.concatenate([original_low, np.full(self._latent_parameter_dim, -np.inf)])
                high = np.concatenate([original_high, np.full(self._latent_parameter_dim, np.inf)])
                new_space = Box(low=low, high=high, dtype=np.float32)

            self.observation_spaces[agent] = new_space

    def _sample_latent_parameter(self) -> None:
        self._latent_parameter = self._gaussian_std * np.random.randn(len(self._env.agents),
                                                                      self._latent_parameter_dim)
        if str(self._env) == 'sc2':
            assert self.num_agents == 6
            # for starcraft env, sync same team member latent param
            self._latent_parameter[1, :] = self._latent_parameter[0, :]
            self._latent_parameter[2, :] = self._latent_parameter[0, :]
            self._latent_parameter[4, :] = self._latent_parameter[3, :]
            self._latent_parameter[5, :] = self._latent_parameter[3, :]

    def _augment_obs(self, obs, agent: str):
        latent_parameter_obs = self._latent_parameter[self._index_map[agent], :]
        if self._use_dict_obs_space:
            joint_obs = {'original_obs': obs, 'random_noise': latent_parameter_obs}
            return joint_obs
        joint_obs = np.concatenate([obs, latent_parameter_obs])
        return joint_obs

    def reset(self) -> None:
        self._sample_latent_parameter()
        obs = self._env.reset()
        return {agent: self._augment_obs(obs[agent], agent) for agent in self.agents}

    def step(self, actions):
        obs, rewards, dones, infos = self._env.step(actions)
        obs = {agent: self._augment_obs(obs[agent], agent) for agent in self.agents}
        return obs, rewards, dones, infos

    def render(self):
            self._env.render()

    def seed(self, seed=None):
        self._env.seed(seed)