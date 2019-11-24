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

from .utils import hard_update

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.FloatTensor)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'ratios', 'entropy'])
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

class CNN_policy(nn.Module):
    def __init__(self, num_actions, input_shape=(3, 84, 84)):
        super(CNN_policy, self).__init__()
        # input N_batch, N_channel, Height, Width
        self.conv1 = nn.Conv2d(3, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)

        self.fc1 = nn.Linear(9 * 9 * 32, 256)
        self.final = nn.Linear(256, num_actions)
        self.critic_value = nn.Linear(256, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x, only_value=False):
        if only_value:
            with torch.no_grad():
                x = torch.from_numpy(x).to(device)
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = x.view(x.size(0), -1)
                x = F.relu(self.fc1(x))
                return self.critic_value(x)
        x = torch.from_numpy(x).to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.final(x)
        value = self.critic_value(x)
        return F.softmax(logits, -1).squeeze(), value.squeeze()
    
    @classmethod
    def get_device(self):
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type(torch.FloatTensor)
        return device

class PPO_Agent():
    def __init__(self, lr=3e-3, clip_value=0.2, policy=Policy, policy_kwargs={}):
        # define policies
        self.policy = policy(**policy_kwargs)
        self.policy_old = policy(**policy_kwargs)
        hard_update(self.policy_old, self.policy)

        self.optimizer = optim.RMSprop(self.policy.parameters(), lr=lr)
        self.eps = np.finfo(np.float32).eps.item()
        self.clip_value=0.2
        self.gamma = 0.99
        self.entropy_bonus = 0.01
        self.policy.train()

    def select_action(self, state):
        # state = torch.from_numpy(state.astype('float32')).to(device)
        probs, state_value = self.policy(state)
        with torch.no_grad():
            probs_old, _ = self.policy_old(state)
        m = Categorical(probs)
        action = m.sample()
        r_t = probs[action]/probs_old[action]
        self.policy.saved_actions.append(SavedAction(m.log_prob(action), state_value, r_t, m.entropy()))
        return action.item()
    
    def get_name(self):
        return "Skynet"

    "GIVEN rewards array from rollout return the returns with zero mean and unit std"        
    def discount_rewards(self, rewards_arr, gamma, start_value=0):
        R = start_value
        returns = []
        for r in rewards_arr[::-1]:
            R = r + R*gamma
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if len(returns) == 1:
            return returns
        return (returns - returns.mean())/(returns.std() + self.eps)

    def compute_single_actor_loss_element(self, r_t, advantage_t):
        first_val = r_t * advantage_t
        second_val = torch.clamp(r_t, 1-self.clip_value, 1+self.clip_value) * advantage_t
        if first_val.item() > second_val.item():
            return second_val 
        else:
            return first_val

    def save_models(self, name="PPO_from_pixels.pth"):
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, name)

    def load_model(self, name="PPO_from_pixels.pth"):
        checkpoint = torch.load(name)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy.train()


    def train_on_rollout(self, writer, T):
        returns = self.discount_rewards(self.policy.rewards, self.gamma)
        actor_loss = []
        critic_loss = []
        entropy_objective = []
        for (log_prob, value, ratio, entropy), r in zip(self.policy.saved_actions, returns):
            advantage = r - value.item()
            # PPO changes the actor loss
            actor_loss.append(-self.compute_single_actor_loss_element(ratio, advantage))
            entropy_objective.append(-entropy)
            critic_loss.append(F.smooth_l1_loss(value, torch.tensor([r])))
        self.optimizer.zero_grad()
        loss = torch.stack(actor_loss).mean() + torch.stack(critic_loss).mean()
        loss += self.entropy_bonus * torch.stack(entropy_objective).mean()
        loss.backward()
        writer.add_scalar("actor and critic loss", loss.item(), T)
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_actions[:]

    def train_on_batch(self, observation, done, writer, gamma, T):
        returns = None
        if done:
            returns = self.discount_rewards(self.policy.rewards, self.gamma, start_value=0)
        else:
            next_state = torch.from_numpy(observation.astype('float32')).to(device).float()
            final_value = self.policy(next_state, only_value=True)
            returns = self.discount_rewards(self.policy.rewards, self.gamma, start_value=final_value.item())
        actor_loss = []
        critic_loss = []
        for (log_prob, value), r in zip(self.policy.saved_actions, returns):
            advantage = r - value.item()
            actor_loss.append(-log_prob * advantage)
            critic_loss.append(F.mse_loss(value, torch.tensor([r])))
        self.optimizer.zero_grad()
        loss_actor = torch.stack(actor_loss).mean()
        writer.add_scalar("actor loss", loss_actor, T)
        loss_critic = torch.stack(critic_loss).mean()
        writer.add_scalar("critic loss", loss_actor, T)
        loss = loss_actor + loss_critic
        loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_actions[:]
    
    def set_old_policy_to_new(self):
        hard_update(self.policy_old, self.policy)


def learn_episodic_A2C(N_eps, max_ep_steps, writer):
    agent = PPO_Agent()
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
            action = agent.select_action(observation)

            observation, reward, done, info = env.step(action)
            agent.policy.rewards.append(reward)
            total_r += reward
            # if done or ((t % batch_update) == 0):
                # train_on_batch(observation, done, writer, df, T)
            if done:
                agent.train_on_rollout(writer, T)
                if initially_updated:
                    agent.set_old_policy_to_new()
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