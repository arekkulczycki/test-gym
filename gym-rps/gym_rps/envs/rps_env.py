import enum
import random

import gym
import numpy


class RPS(enum.Enum):
    ROCK = 1
    PAPER = 2
    SCISSORS = 3

    def __int__(self):
        return self.value


class RockPaperScissors(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(numpy.array([0, 0, 0]), numpy.array([1, 1, 1]))
        self.observation_space = gym.spaces.Discrete(3)
        # self.reward_range =

        self.obs = None
        self.last_obs = None
        self.last_action = None
        self.last_action_result = None
        self.last_reward = None

    def step(self, action):
        action_result = self._take_action(action)
        reward = self._get_reward(action_result)

        self.last_obs = self.obs
        self.obs = self._next_observation()

        self.last_action = action
        self.last_action_result = action_result
        self.last_reward = reward

        return self.obs, reward, True, {}

    def reset(self):
        self.last_obs = None
        self.last_action = None
        self.last_action_result = None
        self.last_reward = None
        return self._next_observation()

    def render(self, mode='human', close=False):
        print('***')
        print(RPS(self.last_obs).name)
        print(self.last_action)
        print(RPS(self.last_action_result).name)
        print(self.last_reward)

    def _next_observation(self):
        self.obs = random.choice([RPS.ROCK, RPS.PAPER, RPS.SCISSORS])
        return self.obs

    def _take_action(self, action):
        return random.choices([RPS.ROCK, RPS.PAPER, RPS.SCISSORS], weights=action)[0]

    def _get_reward(self, action_result):
        if self.obs == RPS.PAPER and action_result == RPS.ROCK:
            return -1
        elif self.obs == RPS.ROCK and action_result == RPS.SCISSORS:
            return -1
        elif self.obs == RPS.SCISSORS and action_result == RPS.PAPER:
            return -1
        elif self.obs == action_result:
            return 0
        else:
            return 1
