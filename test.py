import gym

from stable_baselines.common.policies import CnnLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import wimblepong

# multiprocess environment
env = make_vec_env('WimblepongVisualSimpleAI-v0', n_envs=4)

model = PPO2(CnnLnLstmPolicy, env, \
    verbose=1, policy_kwargs={"ob_space": env.observation_space, "ac_space": env.action_space})
model.learn(total_timesteps=10000)
model.save("ppo2_cartpole")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2_cartpole")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    # env.render()