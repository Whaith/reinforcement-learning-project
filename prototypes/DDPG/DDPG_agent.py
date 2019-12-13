import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal

import numpy as np

from NNets import Q_net, Policy_net
from utils import soft_update, hard_update, ReplayMemory, LinearSchedule, Transition

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MEMORY_SIZE = int(1e6)
GAMMA = 0.99
LR = 1e-3
TAU = 1e-3
WARMUP_STEPS = 10000
E_GREEDY_STEPS = 20000
FINAL_STD = 0.1
BATCH_SIZE = 64

class DDPG_Agent:

    def __init__(self, ob_sp, act_sp, alow, ahigh):
        self.policy = Policy_net(ob_sp, act_sp)
        self.policy_targ = Policy_net(ob_sp, act_sp)
        self.qnet = Q_net(ob_sp, act_sp)
        self.qnet_targ = Q_net(ob_sp, act_sp)

        self.policy.to(device)

        hard_update(self.policy_targ, self.policy)
        hard_update(self.qnet_targ, self.qnet)

        self.p_optimizer = optim.Adam(self.policy.parameters(), lr = LR)
        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr = LR)
        self.memory = ReplayMemory(int(1e6))
        self.epsilon_scheduler = LinearSchedule(E_GREEDY_STEPS, FINAL_STD, 1.0, warmup_steps=WARMUP_STEPS)
        self.n_steps = 0

    def get_action(self, state):
        noise = np.random.normal(0, self.epsilon_scheduler.value(self.n_steps))
        st = torch.from_numpy(state).view(1, -1).float()
        action = self.policy(st)
        self.n_steps += 1
        return action.item() + noise

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        samples = self.memory.sample(min(BATCH_SIZE, len(self.memory)))
        print(len(samples))
        pass

# Q_value_net = Q_net()
# Q_value_net_target = Q_net()
# hard_update(Q_value_net_target, Q_value_net)

# policy_net = Policy_net()
# policy_net_target = Policy_net()
# hard_update(policy_net_target, policy_net)

# Q_value_net_target.eval()
# policy_net_target.eval()

# Q_value_net.to(device)
# Q_value_net_target.to(device)

# policy_net.to(device)
# policy_net_target.to(device)

# optim_q = optim.Adam(Q_value_net.parameters(), lr=1e-3)
# optim_p = optim.Adam(policy_net.parameters(), lr=1e-4)

# def train_on_batch(memory, batch_size, df, T):
#     batch = memory.sample(batch_size)
#     batch_n = Transition(*zip(*batch))
#     actions = torch.tensor(batch_n.action).float().view(-1, 1)
#     states = torch.cat(batch_n.state).float()
#     next_states = torch.cat(batch_n.next_state).float()
#     batch_rewards = torch.cat(batch_n.reward).float().view(-1, 1).to(device)
    
#     dones = torch.tensor(batch_n.done).float().view(-1, 1).to(device)

#     inputs = Q_value_net((states, actions))
#     targets = batch_rewards
#     targets += (1-dones)*df*Q_value_net_target((next_states, policy_net_target(next_states)))
#     loss = F.mse_loss(inputs, targets)
#     # critic loss
#     optim_q.zero_grad()
#     loss.backward()
#     optim_q.step()

#     # actor loss
#     optim_p.zero_grad()
#     loss_actor = - Q_value_net((states, policy_net(states))).mean()
#     loss_actor.backward()
#     optim_p.step()

#     soft_update(Q_value_net, Q_value_net_target, tau=0.001)
#     soft_update(policy_net, policy_net_target, tau=0.001)