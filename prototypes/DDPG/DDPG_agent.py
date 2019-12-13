import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal

from NNets import Q_net, Policy_net
from utils import soft_update, hard_update, ReplayMemory, LinearSchedule

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MEMORY_SIZE = int(1e6)
GAMMA = 0.99
LR = 1e-3
TAU = 1e-3
WARMUP_STEPS = 10000
BATCH_SIZE = 64

class DDPG_Agent:

    def __init__(self, ob_sp, act_sp, alow, ahigh):
        self.policy = Policy_net(ob_sp, act_sp)
        self.policy_targ = Policy_net(ob_sp, act_sp)
        self.qnet = Q_net(ob_sp, act_sp)
        self.qnet_targ = Q_net(ob_sp, act_sp)

        hard_update(self.policy_targ, self.policy)
        hard_update(self.qnet_targ, self.qnet)

        self.p_optimizer = optim.Adam(self.policy.parameters(), lr = LR)
        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr = LR)
        self.memory = ReplayMemory(int(1e6))

    def act(self, state):
        
        pass

    def train(self):
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