import gym, roboschool
import numpy as np


from pettingzoo.utils.env import AECEnv

class RoboschoolAntWrapper(AECEnv):
    def __init__(self) -> None:
        self._env = gym.make("RoboschoolAnt-v1")
        self.possible_agents = ['front_legs', 'rear_legs']
        self.agents = self.possible_agents
        self._obs = {name: None for name in self.agents}
        self.rewards = {name: 0 for name in self.agents}
        self.dones = {name: False for name in self.agents}
        self.infos = {name: None for name in self.agents}
        self.metadata = self._env.metadata
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.observation_spaces = {'front_legs': self._env.observation_space,
                                   'rear_legs': self._env.observation_space}
        original_action_space = self._env.action_space
        new_action_space = gym.spaces.box.Box(low=original_action_space.low[:4],
                                              high=original_action_space.high[:4],
                                              dtype=original_action_space.dtype)
        self.action_spaces = {'front_legs': new_action_space,
                              'rear_legs': new_action_space}

    def reset(self) -> None:
        obs = self._env.reset()
        # attention: self.agents is important to be used by the pettingzooEnv Wrapper
        self.agents = self.possible_agents
        self.dones = {'front_legs': False, 'rear_legs': False, '__all__': False}
        for key in self.agents:
            self._obs[key] = obs
        self.agent_selection = 'front_legs'
        return self._obs

    def observe(self, agent):
        obs = self._obs[agent]
        assert obs is not None
        return obs

    def step(self, action):
        if len(self.agents) == 0:
            return
        if self.agent_selection == 'front_legs':
            self._action_front = action
            self.agent_selection = 'rear_legs'
        else:
            assert self.agent_selection == 'rear_legs' and self._action_front is not None
            self._action_rear = action
            action = np.concatenate([self._action_rear[-2:], self._action_front, self._action_rear[:2]])
            obs, reward, done, info = self._env.step(action)
            self._obs = {'front_legs': obs, 'rear_legs': obs}
            self.rewards = {'front_legs': reward, 'rear_legs': -reward}
            self.dones = {'front_legs': done, 'rear_legs': done, '__all__': done}
            for agent in self.agents:
                self._cumulative_rewards[agent] += self.rewards[agent]
            self.agent_selection = 'front_legs'
        self.agents = [agent for agent in self.agents if not self.dones[agent]]
        return self._obs, self.rewards, self.dones, self.infos

    def render(self, mode):
        self._env.render(mode)

    def seed(self, seed=None):
        self._env.seed(seed)