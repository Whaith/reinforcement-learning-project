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

#%%
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


SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'ratios', 'entropy'])

# define policies
policy = Policy()
policy_old = Policy()
hard_update(policy_old, policy)

optimizer = optim.RMSprop(policy.parameters(), lr=3e-3)
eps = np.finfo(np.float32).eps.item()

def select_action(state, old_policy):
    state = torch.from_numpy(state).float()
    probs, state_value = policy(state)
    with torch.no_grad():
        probs_old, _ = policy_old(state)
    m = Categorical(probs)
    action = m.sample()
    r_t = probs[action]/probs_old[action]
    policy.saved_actions.append(SavedAction(m.log_prob(action), state_value, r_t, m.entropy()))
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
    return (returns - returns.mean())/(returns.std() + eps)

def compute_single_actor_loss_element(r_t, advantage_t, clip_value=0.2):
    first_val = r_t * advantage_t
    second_val = torch.clamp(r_t, 1-clip_value, 1+clip_value) * advantage_t
    if first_val.item() > second_val.item():
        return second_val 
    else:
        return first_val

def train_on_rollout(gamma, writer, T):
    returns = discount_rewards(policy.rewards, gamma)
    actor_loss = []
    critic_loss = []
    entropy_objective = []
    for (log_prob, value, ratio, entropy), r in zip(policy.saved_actions, returns):
        advantage = r - value.item()
        # PPO changes the actor loss
        actor_loss.append(-compute_single_actor_loss_element(ratio, advantage))
        entropy_objective.append(-entropy)
        critic_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    loss = torch.stack(actor_loss).mean() + torch.stack(critic_loss).mean()
    loss += 0.1 * torch.stack(entropy_objective).mean()
    loss.backward()
    writer.add_scalar("actor and critic loss", loss.item(), T)
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]

def train_on_batch(observation, done, writer, gamma, T):
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
    loss_actor = torch.stack(actor_loss).mean()
    writer.add_scalar("actor loss", loss_actor, T)
    loss_critic = torch.stack(critic_loss).mean()
    writer.add_scalar("critic loss", loss_actor, T)
    loss = loss_actor + loss_critic
    loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]

def learn_episodic_A2C(N_eps, max_ep_steps, writer):
    df = 0.99
    rewards = []
    env = gym.make('CartPole-v0')
    batch_update = 20
    env._max_episode_steps = max_ep_steps
    T = 0
    initially_updated = False
    for i_episode in range(N_eps):
        observation = env.reset()
        total_r = 0
        for t in range(100000):
            action = select_action(observation, policy_old)

            observation, reward, done, info = env.step(action)
            policy.rewards.append(reward)
            total_r += reward
            # if done or ((t % batch_update) == 0):
                # train_on_batch(observation, done, writer, df, T)
            if done:
                train_on_rollout(df, writer, T)
                if initially_updated:
                    hard_update(policy_old, policy)
                initially_updated = True
                if (i_episode + 1) % 100 == 0:                
                    print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                break
            T += 1
        writer.add_scalar("Ep reward", total_r, i_episode)

    # writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    env.close()
    return rewards
# N_EPS = 2000
# rewards_A2C = learn_episodic_A2C(N_EPS, 500)

if __name__ == '__main__':
    writer = SummaryWriter()
    learn_episodic_A2C(500, 500, writer)