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
from tensorboardX import SummaryWriter

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x, only_value=False):
        if only_value:
            with torch.no_grad():
                x = F.relu(self.affine1(x))
                state_values = self.value_head(x)
                return state_values

        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

policy = Policy()
optimizer = optim.RMSprop(policy.parameters(), lr=3e-3)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()

"GIVEN rewards array from rollout return the returns with zero mean and unit std"        
def discount_rewards(rewards_arr, gamma, start_value=0):
    R = start_value
    returns = []
    for r in rewards_arr[::-1]:
        R = r + R*gamma
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if len(returns) == 1:
        return returns
    else:
        return returns

def train_on_rollout(gamma=0.99):
    returns = discount_rewards(policy.rewards, gamma)
    actor_loss = []
    critic_loss = []
    for (log_prob, value), r in zip(policy.saved_actions, returns):
        advantage = r - value.item()
        actor_loss.append(-log_prob * advantage)
        critic_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
    loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]

def train_on_batch(observation, done, gamma=0.99):
    returns = None
    if done:
        returns = discount_rewards(policy.rewards, gamma, start_value=0)
    else:
        next_state = torch.from_numpy(observation).float()
        final_value = policy(next_state, only_value=True)
        returns = discount_rewards(policy.rewards, gamma, start_value=final_value.item())
    actor_loss = []
    critic_loss = []
    for (log_prob, value), r in zip(policy.saved_actions, returns):
        advantage = r - value.item()
        actor_loss.append(-log_prob * advantage)
        critic_loss.append(F.mse_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    loss = torch.stack(actor_loss).sum() + torch.stack(critic_loss).sum()
    loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]
        

def learn_episodic_A2C(N_eps=500, max_ep_steps=500):
    df = 0.99
    rewards = []
    env = gym.make('CartPole-v0')
    batch_update = 20
    env._max_episode_steps = max_ep_steps
    for i_episode in range(N_eps):
        observation = env.reset()
        total_r = 0
        for t in range(100000):
            action = select_action(observation)
            observation, reward, done, info = env.step(action)
            policy.rewards.append(reward)
            total_r += reward
            if done or ((t % batch_update) == 0):
                train_on_batch(observation, done, df)
            if done:
                # train_on_rollout(0.99)
                if (i_episode + 1) % 100 == 0:                
                    print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                break
        rewards.append(total_r)
    env.close()
    return rewards
# N_EPS = 2000
# rewards_A2C = learn_episodic_A2C(N_EPS, 500)

if __name__ == '__main__':

    learn_episodic_A2C(500, 500)