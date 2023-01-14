import gym, roboschool
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

class RoboschoolAntWrapper(MultiAgentEnv):
    def __init__(self) -> None:
        self._env = gym.make("RoboschoolAnt-v1")
        self.agents = ['front_legs', 'rear_legs']
        self.metadata = self._env.metadata
        self.observation_spaces = {'front_legs': self._env.observation_space,
                                   'rear_legs': self._env.observation_space}
        original_action_space = self._env.action_space
        new_action_space = gym.spaces.box.Box(low=original_action_space.low[:4],
                                              high=original_action_space.high[:4],
                                              dtype=original_action_space.dtype)
        self.action_spaces = {'front_legs': new_action_space,
                              'rear_legs': new_action_space}
        self.observation_space = self.observation_spaces['front_legs']
        self.action_space = self.action_spaces['front_legs']

    def reset(self):
        obs = self._env.reset()
        return {name: obs for name in self.agents}

    def step(self, actions):
        front_action = actions['front_legs']
        rear_action = actions['rear_legs']
        action = np.concatenate([rear_action[-2:], front_action, rear_action[:2]])
        obs, reward, done, info = self._env.step(action)
        reward = reward / 10.0
        obs = {'front_legs': obs, 'rear_legs': obs}
        rewards = {'front_legs': reward, 'rear_legs': -reward}
        dones = {'front_legs': done, 'rear_legs': done, '__all__': done}
        infos = {name: info for name in self.agents}
        return obs, rewards, dones, infos

    def render(self):
        self._env.render()

    def seed(self, seed=None):
        self._env.seed(seed)