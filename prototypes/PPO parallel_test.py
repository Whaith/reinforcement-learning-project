#%%
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
from tensorboardX import SummaryWriter

from utils import hard_update

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.FloatTensor)
#%%
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, only_value=False):
        if only_value:
            with torch.no_grad():
                x = F.relu(self.affine1(x))
                state_value = self.value_head(x)
                return state_value
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type(torch.FloatTensor)
    return device

class PPO_Agent():
    def __init__(self, lr=3e-3, clip_value=0.2, \
            policy=Policy, policy_kwargs={}, df=0.99, gradient_clip=0.5):
        # define policies
        self.policy = policy(**policy_kwargs)
        self.policy_old = policy(**policy_kwargs)
        self.initially_updated = False
        self.device = get_device()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-3)
        self.eps = np.finfo(np.float32).eps.item()
        self.gradient_clip = gradient_clip
        self.clip_value=0.2
        self.gamma = df
        self.n_epochs = 3
        self.c1 = 1.0
        self.c2 = 0.01
        self.GAE = 0.95
        self.mse = nn.MSELoss()

        self.policy.train()
        self.old_action_logprobs = []
        self.old_actions = []
        self.states = []
        self.rewards = []
        self.r_t_s = []
        self.dones = []

    def to_tensor(self, ndarr):
        return torch.from_numpy(ndarr.astype('float32')).to(self.device)

    def select_action(self, state):
        state = self.to_tensor(state)
        probs, state_value = self.policy_old(state)
        m = Categorical(probs)
        action = m.sample()
        self.states.append(state)
        self.old_action_logprobs.append(m.log_prob(action).item())
        self.old_actions.append(action)
        return action.item()

    def get_name(self):
        return "Skynet"

    def set_old_policy_to_new(self):
        hard_update(self.policy_old, self.policy)
    
    "GIVEN rewards array from rollout return the returns with zero mean and unit std"        
    def discount_rewards(self, rewards_arr, start_value=0):
        R = start_value
        returns = []
        # if len(rewards_arr) == 0:
        #     print(1)
        #     print(2)
        for i in range(len(rewards_arr) - 1, -1, -1):
            r = rewards_arr[i]
            done = self.dones[i]
            if done:
                R = 0
            R = r + R*self.gamma
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if len(returns) == 1:
            return returns
        else:
            return returns

    def compute_advantages(self, rewards, values, next_state_values, dones_mask):
        td_targets = rewards + self.gamma * next_state_values
        td_errors = td_targets - values.detach()
        advantages = []
        start_value = 0
        for i in range(len(td_errors)-1, -1, -1):
            if dones_mask[i]:
                start_value = 0
            start_value = td_errors[i].item() + self.gamma * self.GAE * start_value
            advantages.insert(0, start_value)
        advantages = torch.tensor(advantages)
        # advantages = (advantages - advantages.mean())/(advantages.std() + 1e-6)
        return td_targets, advantages

    def get_state_values_and_probs(self, old_actions, old_states):
        probs, state_values = self.policy(old_states)
        m = Categorical(probs)
        new_logprobs = m.log_prob(old_actions)
        return new_logprobs, state_values.view(-1), m.entropy()


    def train_on_batch(self, observation, done, GAE=0.95, df=0.99):
        # assert len(self.states) == 10, "qqqq"
        if done:
            final_value = torch.zeros(1)
        else:
            final_value = self.policy(self.to_tensor(observation), only_value=True)
        # rewards r_t to r_T + 1
        rewards = torch.tensor(self.rewards).squeeze(-1).to(device)
        
        # logprobs of old policy

        old_policy_logprobs = torch.tensor(self.old_action_logprobs).to(device).detach()
        old_actions = torch.stack(self.old_actions).detach()
        old_states = torch.stack(self.states).detach()
        # pritn(self.n_epochs)
        for i in range(1):
            #   1. calculate new state_values, next_state_values, advantages:
            #   2. calculate new ratio nominators (for cur policy)
            #   3. calculate new entropies
            new_logprobs, new_values, entropy = self.get_state_values_and_probs(old_actions, old_states)

            try:
                next_state_values = torch.cat([new_values[1:].detach(), final_value])
            except:
                pass
            # calculate new logprobs of an for a current policy
            value_targets, advantages = self.compute_advantages(rewards, new_values,\
                next_state_values, self.dones)
            # print(advantages)
            # rewards = torch.tensor(rewards).to(device)
            # returns = self.discount_rewards(rewards)
            # returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            # advantages = returns - new_values.detach()
            # ratio terms: 

            ratios = torch.exp(new_logprobs - old_policy_logprobs.detach())

            # surrogate_loss:
            surr_1 = ratios * advantages
            surr_2 = ratios.clamp(1 - self.clip_value, 1 + self.clip_value) * advantages
            L_clip = torch.min(ratios * advantages, \
                ratios.clamp(1 - self.clip_value, 1 + self.clip_value) * advantages)
            # print('hereeeeeee')
            L_vf = self.mse(new_values.view(-1, 1), value_targets.view(-1, 1))
            loss_total = -(L_clip + self.c2 * entropy).mean() + L_vf
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

        self.old_action_logprobs = []
        self.old_actions = []
        self.states = []
        self.rewards = []
        self.r_t_s = []
        self.dones = []

        
        self.policy_old.load_state_dict(self.policy.state_dict())

        

def learn_episodic_A2C(N_eps, max_ep_steps, writer):
    agent = PPO_Agent(Policy)
    df = 0.99
    rewards = []
    env = gym.make('CartPole-v0')
    env._max_episode_steps = max_ep_steps
    T = 1
    T_max = 100
    for i_episode in range(N_eps):
        observation = env.reset()
        total_r = 0
        done = False
        t = 0
        while not done:
            action = agent.select_action(observation)
            observation, reward, done, info = env.step(action)
            agent.rewards.append(reward)
            agent.dones.append(done)
            total_r += reward
            if T % T_max == 0:
                # train on batch
                agent.train_on_batch(observation, done)
            if done:
                # agent.train_on_rollout(writer, T)
                if (i_episode + 1) % 100 == 0:                
                    print("Episode {} finished after {} timesteps".format(i_episode, total_r))
            T += 1
        # writer.add_scalar("Ep reward", total_r, i_episode)

    # writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    env.close()
    return rewards
# N_EPS = 2000ini
# rewards_A2C = learn_episodic_A2C(N_EPS, 500)

if __name__ == '__main__':
    writer = SummaryWriter()
    learn_episodic_A2C(5000, 500, writer)