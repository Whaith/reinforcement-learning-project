import sys
sys.path.append('.')

import os
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
import cv2
import torch.multiprocessing as mp
import torch.distributed as dist
import wimblepong


T_HORIZON = 128
ADAM_LR = 2.5e-4 # TODO ANNEAL this over the training to 0
N_EPOCHS = 3
DF = 0.989
N_ACTORS = 8
CLIP_PARAM = 0.1 # TODO ANNEAL this over during training to 0
ENTROPY_C2 = 0.01
MEM_SIZE = 256
IMAGE_H_W = 80

def image_to_grey(obs, target_reso=(80, 80)):
    return (np.dot(cv2.resize(obs[...,:3], dsize=target_reso), \
        [0.2989, 0.5870, 0.1140]).astype('float32')/255.0 + 0.15).round()

# source https://github.com/greydanus/baby-a3c/blob/master/baby-a3c.py
class NNPolicy(nn.Module): # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.gru = nn.GRUCell(32 * 5 * 5, 256)
        self.critic_linear, self.actor_linear = nn.Linear(256, 1), nn.Linear(256, num_actions)

    def forward(self, inputs, train=True, hard=False):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        hx = self.gru(x.view(-1, 32 * 5 * 5), (hx))
        return F.softmax(self.actor_linear(hx), -1), self.critic_linear(hx), hx


class PPO_Agent():

    def __init__(self, policy):
        self.gamma = DF
        self.eps = np.finfo(np.float32).eps.item()
        self.batch_update_freq = T_HORIZON
        
        self.policy = policy

        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.states = []
        self.hiddens = []
        
    def select_action(self, state):
        state, hx_0  = state
        state = torch.from_numpy(state).view(1, 1, IMAGE_H_W, IMAGE_H_W)
        probs, _, hx = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.actions.append(action.item())
        self.logprobs.append(m.log_prob(action).item())
        self.states.append(state)
        self.hiddens.append(hx_0)

        return action.item(), hx

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
        return (returns - returns.mean())/(returns.std() + self.eps)

    def get_experience(self, final_obs=None, done=True):
        state, hx = final_obs
        state = torch.from_numpy(state).view(1, 1, IMAGE_H_W, IMAGE_H_W)
        _, state_value, _ = self.policy((state, hx))
        final_value = state_value.detach() if not done else 0.0

        # rewards
        returns = self.discount_rewards(self.rewards, \
            self.dones, self.gamma, final_value)
        
        states = torch.stack(self.states).float()
        old_actions = self.actions
        old_logprobs = torch.tensor(self.logprobs).float()
        hiddens = torch.stack(self.hiddens).float()
        # print('collected experience')
        return states, torch.tensor(old_actions), old_logprobs, returns, hxs

    def clear_experience(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        dist.barrier()

class PPO_Centralized_Trainer():
    def __init__(self, policy_shared, policy):
        self.shared_policy = shared_policy
        self.policy = policy
        self.clip_val = CLIP_PARAM
        self.c2 = ENTROPY_C2
        self.optimizer = optim.Adam(self.policy.parameters(), lr=ADAM_LR)
        self.n_epochs = N_EPOCHS

    def train(self, states, old_actions, old_logprobs, returns, hiddens):

        # print('processing experience')
        for i in range(self.n_epochs):
            # Calculate needed values    
            p, v, _ = policy.forward(zip(states, hiddens))
            m = Categorical(p)
            c = m.log_prob(old_actions)
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

        self.shared_policy.load_state_dict(self.policy.state_dict())

        dist.barrier()

def run(shared_policy, policy, rank, size):
    N_eps = 1e6
    group = dist.new_group([i for i in range(size)])
    batch_update_freq = T_HORIZON
    # Agent getting the training data
    if rank != 0:

        if rank == 1: rewards = []
        env = gym.make("WimblepongVisualSimpleAI-v0")
        action_space = env.action_space.n
        env.seed(rank) ; torch.manual_seed(rank) ; np.seed(rank) # seed everything

        T = 0
        agent = PPO_Agent(shared_policy)

        for i_episode in range(N_eps):
            observation = env.reset()
            if rank == 1: total_r = 0
            done = True
            hx = torch.zeros(1, 256) if done else hx.detach()
            while True:
                T += 1

                action = agent.select_action((image_to_grey(observation), hx))
                observation, reward, done, info = env.step(action)
                observation = image_to_grey(observation)

                agent.rewards.append(reward)
                agent.dones.append(done)

                if rank == 1: total_r += reward
                if T % batch_update_freq == 0:

                    a,b,c,d,h = agent.get_experience(observation, done)
                    dist.gather(a, gather_list=[], dst=0, group=group)
                    # dist.barrier()
                    dist.gather(b, gather_list=[], dst=0, group=group)
                    # dist.barrier()
                    dist.gather(c, gather_list=[], dst=0, group=group)
                    # # dist.barrier()
                    dist.gather(d, gather_list=[], dst=0, group=group)
                    
                    dist.gather(h, gather_list=[], dst=0, group=group)
                    
                    agent.clear_experience()
                if done:
                    print(f"rank: {rank}, episode: {i_episode}")
                    if (i_episode + 1) % 100 == 0:                
                        if rank == 1: print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                    break
            if rank == 1: rewards.append(total_r)
        env.close()
    # centralized actor training on the training data
    else:
        trainer = PPO_Centralized_Trainer(shared_policy, policy)
        old_states = [torch.zeros((30, IMAGE_H_W, IMAGE_H_W), dtype=torch.float32) for i in range(size)]
        old_actions = [torch.zeros((30), dtype=torch.int64) for i in range(size)]
        old_logprobs = [torch.zeros((30), dtype=torch.float32) for i in range(size)]
        old_returns = [torch.zeros((30), dtype=torch.float32) for i in range(size)]
        old_hiddens = [torch.zeros((30, MEM_SIZE), dtype=torch.float32) for i in range(size)]

        while(True):
            dist.gather(old_states[0], gather_list=old_states, dst=0, group=group)
            dist.gather(old_actions[0], gather_list=old_actions, dst=0, group=group)
            dist.gather(old_logprobs[0], gather_list=old_logprobs, dst=0, group=group)
            dist.gather(old_returns[0], gather_list=old_returns, dst=0, group=group)
            dist.gather(old_hiddens[0], gather_list=old_hiddens, dst=0, group=group)
            states = torch.cat(old_states[1:])
            actions = torch.cat(old_actions[1:])
            logprobs = torch.cat(old_logprobs[1:])
            returns = torch.cat(old_returns[1:])
            hiddens = torch.cat(old_hiddens[1:])
            trainer.train(states, actions, logprobs, returns, hiddens)


def init_process(shared_policy, policy, rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '30025'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['GLOO_SOCKET_IFNAME'] = 'eno1'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(shared_policy, policy, rank, size)

if __name__ == '__main__':
    num_agents = N_ACTORS
    # one thread 0 is reserved for centralized learner
    env = gym.make("WimblepongVisualSimpleAI-v0")

    shared_policy = NNPolicy(1, MEM_SIZE, env.action_space.n)
    shared_policy.share_memory()
    policy = NNPolicy(1, MEM_SIZE, env.action_space.n)
    policy.load_state_dict(shared_policy.state_dict())

    size = num_agents + 1
    processes = []
    for rank in range(size):
        p = mp.Process(target=init_process, args=(shared_policy, policy, rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


