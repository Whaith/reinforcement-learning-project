import gym

from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines import DDPG

model = DDPG(DDPGPolicy, 'Pendulum-v0', verbose=1)
# Train the agent
model.learn(total_timesteps=100000)
