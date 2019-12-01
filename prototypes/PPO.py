import sys
sys.path.append('.')

import gym
import numpy as np
from itertools import count
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import random
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# from utils import RAdam

def moving_average(x, N):
    return np.convolve(x, np.ones(N, ), mode='valid') / N

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.states = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values.squeeze()

policy = Policy()
policy_old = Policy()
policy_old.load_state_dict(policy.state_dict())

optimizer = optim.RMSprop(policy.parameters(), lr=3e-3)
# optimizer = RAdam(policy.parameters(), lr=3e-3)
eps = np.finfo(np.float32).eps.item()

# def reset_globals():
#     global policy
#     global policy_old
#     global optimizer
    
    # policy = Policy()
    # policy_old = Policy()
    # policy_old.load_state_dict(policy.state_dict())
    
    # optimizer = optim.RMSprop(policy.parameters(), lr=3e-3)

class PPO_Agent():

    def __init__(self):
        self.gamma = 0.99
        self.batch_update_freq = 300
        self.clip_val = 0.2
        self.c2 = 0.0001
        self.n_epochs = 4

        self.policy = Policy()
        self.policy_old = Policy()
        self.policy_old.load_state_dict(self.policy.state_dict())
    
        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=3e-3)
        
    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.policy_old(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.actions.append(action.item())
        self.policy.logprobs.append(m.log_prob(action).item())
        self.policy.states.append(state)
    #     policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
        return action.item()

    "GIVEN rewards array from rollout return the returns with zero mean and unit std"        
    def discount_rewards(self, rewards_arr, dones, gamma, final_value=0):
        R = final_value
        returns = []
        zipped = list(zip(rewards_arr, dones))
        for (r, done) in zipped[::-1]:
            if done:
                R = 0
            R = r + R*gamma
            returns.insert(0, R)
        returns = torch.tensor(returns)
        return (returns - returns.mean())/(returns.std() + eps)

    def batch_update_agent(self, final_obs=None, done=True):
        state = torch.from_numpy(final_obs).float()
        _, state_value = self.policy(state)
        final_value = state_value.detach() if not done else 0.0
        returns = self.discount_rewards(self.policy.rewards, \
            self.policy.dones, self.gamma, final_value)
        
        states = torch.stack(self.policy.states).float()
    #     print(states.shape)
        old_actions = self.policy.actions
    #     print(old_actions)
        old_logprobs = torch.tensor(self.policy.logprobs).float()
    #     print(old_logprobs.shape)
        
        # PPO OLD VALUES
        for i in range(self.n_epochs):
            # Calculate needed values    
            p, v = self.policy.forward(states)
            m = Categorical(p)
            c = m.log_prob(torch.tensor(old_actions))
            entr = m.entropy()

            # value fn loss
            loss_vf = F.mse_loss(v, returns)

            # surrogate loss
            advantage = returns - v.detach()
            r_ts = torch.exp(c - old_logprobs)
            loss_surr = - (torch.min(r_ts * advantage, \
                torch.clamp(r_ts, 1-self.clip_val, 1+self.clip_val) * advantage)).mean()
            
            # maximize entropy bonus
            loss_entropy = - self.c2 * entr.mean()

            # the total_loss
            loss_total = loss_vf + loss_surr + loss_entropy
            
            # step
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())    
        
        del self.policy.actions[:]
        del self.policy.states[:]
        del self.policy.logprobs[:]
        del self.policy.rewards[:]
        del self.policy.dones[:]

def learn_PPO_single_threaded(N_eps=500, max_ep_steps=500):
    rewards = []
    env = gym.make('CartPole-v0')
    env._max_episode_steps = max_ep_steps
    T = 0

    agent = PPO_Agent()
    
    for i_episode in range(N_eps):
        observation = env.reset()
        total_r = 0
        for t in range(100000):
            T += 1
            action = agent.select_action(observation)
            observation, reward, done, info = env.step(action)

            agent.policy.rewards.append(reward)
            agent.policy.dones.append(done)

            total_r += reward
            if T % agent.batch_update_freq == 0:
                agent.batch_update_agent(observation, done=True)
            if done:
#                 train_on_batch(0.99, observation, done)
                if (i_episode + 1) % 100 == 0:                
                    print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                break
        rewards.append(total_r)
    env.close()
    return rewards

N_EPS = 1500

rewards_PPO = learn_PPO_single_threaded(N_EPS, 500)
plt.plot(moving_average(rewards_PPO, 150), label='RMSprop')

# Try out radam
# reset_globals()

# # optimizer = RAdam(policy.parameters(), lr=3e-3)
# rewards_PPO_RAdam = learn_PPO_single_threaded(N_EPS, 500)
# plt.plot(moving_average(rewards_PPO_RAdam, 150), label='RAdam')
# plt.legend()
# plt.show()