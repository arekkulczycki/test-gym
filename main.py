import gym
from gym import envs
from gym_rps.envs import RockPaperScissors

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

env = gym.make('rps-v0')
obs = env.reset()

total_reward = 0
for i in range(10):
    obs, reward, finished, info = env.step(env.action_space.sample())
    total_reward = total_reward + reward
    env.render()

print('Total Reward: ', total_reward)

# env = DummyVecEnv([lambda: RockPaperScissors()])
# env = gym.make('rps-v0')
# model = PPO2(MlpPolicy, env, learning_rate=0.001)
# model.learn(10000)
# model.save('./rps_model.v0')

model = PPO2.load('./rps_model.v0')
total_reward = 0
for i in range(10):
    action, _states = model.predict(obs)
    obs, reward, dones, info = env.step(action)
    total_reward = total_reward + reward
    env.render()
print('Total Reward: ', total_reward)
