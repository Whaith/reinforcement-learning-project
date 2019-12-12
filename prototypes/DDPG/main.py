import gym
import numpy as np
from itertools import count
from collections import namedtuple
import matplotlib.pyplot as plt
import random
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print(device)

def moving_average(x, N):
    return np.convolve(x, np.ones(N, ), mode='valid') / N

# taken from openAI baselines
class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

class Policy_net(nn.Module):
    def __init__(self):
        super(Policy_net, self).__init__()
        self.affine1 = nn.Linear(3, 200)
        self.affine2 = nn.Linear(200, 100)
        self.mean_head = nn.Linear(100, 1)
        # self.sigma = torch.nn.Parameter(torch.tensor([self.sigma], requires_grad=True))

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        # -2 to 2 with tanh
        mean = 2*torch.tanh(self.mean_head(x))
        return  mean

class Q_net(nn.Module):
    def __init__(self):
        super(Q_net, self).__init__()
        action_space = 1
        self.affine1 = nn.Linear(3, 200)
        self.affine2 = nn.Linear(200, 100)
        self.value_head = nn.Linear(100, 10)
        self.value_head2 = nn.Linear(action_space + 10, 1)

    def forward(self, state):
        x, action  = state
        x, action = x.to(device), action.to(device)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.value_head(x))
        x = torch.cat([x, action], 1)
        return self.value_head2(x)
    
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# https://github.com/WenhangBao/Multi-Agent-RL-for-Liquidation/blob/master/ddpg_agent.py
def soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    Params
    ======
        local_model: PyTorch model (weights will be copied from)
        target_model: PyTorch model (weights will be copied to)
        tau (float): interpolation parameter 
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

# https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
        
Q_value_net = Q_net()
Q_value_net_target = Q_net()
hard_update(Q_value_net_target, Q_value_net)

policy_net = Policy_net()
policy_net_target = Policy_net()
hard_update(policy_net_target, policy_net)

Q_value_net_target.eval()
policy_net_target.eval()

Q_value_net.to(device)
Q_value_net_target.to(device)

policy_net.to(device)
policy_net_target.to(device)

optim_q = optim.Adam(Q_value_net.parameters(), lr=1e-3)
optim_p = optim.Adam(policy_net.parameters(), lr=1e-4)

def train_on_batch(memory, batch_size, df, T):
    # TODO-in future: remove the casting to tensors all the time
    # Vectorized implementation
    batch = memory.sample(batch_size)
    # connect all batch Transitions to one tuple
    batch_n = Transition(*zip(*batch))
    # reshape actions so ve can collect the DQN(S_t, a_t) easily with gather
    actions = torch.tensor(batch_n.action).float().view(-1, 1)
    # get batch states
#     print(batch_n.state)
    states = torch.cat(batch_n.state).float()
    next_states = torch.cat(batch_n.next_state).float()
    batch_rewards = torch.cat(batch_n.reward).float().view(-1, 1).to(device)
    
    dones = torch.tensor(batch_n.done).float().view(-1, 1).to(device)

    inputs = Q_value_net((states, actions))
    targets = batch_rewards
    targets += (1-dones)*df*Q_value_net_target((next_states, policy_net_target(next_states)))
    loss = F.mse_loss(inputs, targets)
    # critic loss
    optim_q.zero_grad()
    loss.backward()
    optim_q.step()

    # actor loss
    # before = Q_value_net((states, policy_net(states))).mean()
    optim_p.zero_grad()
    loss_actor = - Q_value_net((states, policy_net(states))).mean()
#     writer.add_scalar("Actor loss", loss_actor.item(), T)
    loss_actor.backward()
    optim_p.step()
    # after = Q_value_net((states, policy_net(states))).mean()
    # try:
    # assert before < after
    # except:
        # pritn(1)

    soft_update(Q_value_net, Q_value_net_target, tau=0.001)
    soft_update(policy_net, policy_net_target, tau=0.001)

# class OUNoise:
#     """Ornstein-Uhlenbeck process."""

#     def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
#         """Initialize parameters and noise process."""
#         self.mu = mu * np.ones(size)
#         self.theta = theta
#         self.sigma = sigma
#         self.seed = random.seed(seed)
#         self.reset()

#     def reset(self):
#         """Reset the internal state (= noise) to mean (mu)."""
#         self.state = copy.copy(self.mu)

#     def sample(self):
#         """Update internal state and return it as a noise sample."""
#         x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
#         self.state = x + dx
#         return self.state

    
def learn_episodic_DDPG(N_eps=500): 
    
    memory_len = int(1e5)
    df = 0.99
    batch_size = 128
    train_freq = 1
    T = 0
    # target_update_freq = 1000
    
    # scheduler
    e_s = 1.0
    e_e = 0.01
    N_decay = 100000
    scheduler = LinearSchedule(N_decay, e_e, e_s)
    
    # replay mem
    memory = ReplayMemory(memory_len)
    rewards = []
#     writer = SummaryWriter()

    env = gym.make('Pendulum-v0')
    # n_actions = env.action_space.n
    noise = 0
    actions = []
    from collections import deque

    running_epr = deque([0 for i in range(101)],  maxlen=100)
    for i_episode in range(N_eps):
        
        observation = env.reset()
        total_r = 0
        done = False

        for t in range(300):
            T += 1

            curr_epsilon = scheduler.value(T)
            noise = np.random.normal(0, curr_epsilon)
            action_mean = policy_net(torch.from_numpy(observation).float())
            action = np.clip(action_mean.item() + noise , -1, 1)

                # print(action)
            next_observation, reward, done, info = env.step([action])
            total_r += reward
            reward = torch.tensor([reward])
            
            memory.push(torch.from_numpy(observation).view(1, -1), \
                action, reward, torch.from_numpy(next_observation).view(1, -1), float(done))
            
            # train the DQN
            if T % train_freq == 0:
                train_on_batch(memory, min(batch_size, T), df, T)

            observation = next_observation

            if done:
                break
        print("done episode ", i_episode)
        # print(np.mean(running_epr))
        # writer.add_scalar("Episode_reward", total_r, i_episode)
        running_epr.append(total_r)
        if (i_episode + 1) % 100 == 0:
            print('curr eps', noise, "epsilon", curr_epsilon)
            print("Episode {} finished with {} total rewards, T: {}".format(i_episode, np.mean(running_epr), T))
                
        rewards.append(total_r)

    # render environment
    for i in range(5):
        observation = env.reset()
        for j in range(500):
            action_mean = policy_net(torch.from_numpy(observation).float())
            action = np.clip(action_mean.item(), -2.0, 2.0)
            next_observation, reward, done, info = env.step([action])
            env.render()
    env.close()
    
    return rewards

N_EPS = 10000
# rewards_DQN_dueling = learn_episodic_DQN(N_EPS, 500, use_dueling=True)
rewards_DDPG = learn_episodic_DDPG(N_EPS)
plt.plot(moving_average(rewards_DDPG, 100), label="DDPG")
plt.legend()
plt.show()
    

