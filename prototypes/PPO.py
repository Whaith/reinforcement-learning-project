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
import os
import torch.multiprocessing as mp
import torch.distributed as dist

def moving_average(x, N):
    return np.convolve(x, np.ones(N, ), mode='valid') / N

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values.squeeze()

# policy = Policy()
# policy_old = Policy()
# policy_old.load_state_dict(policy.state_dict())

# optimizer = optim.RMSprop(policy.parameters(), lr=3e-3)
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

    def __init__(self, policy):
        self.gamma = 0.99
        self.batch_update_freq = 30

        self.policy = policy
        # self.policy_old = Policy()
        # self.policy_old.load_state_dict(self.policy.state_dict())

        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.states = []
        
    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.actions.append(action.item())
        self.logprobs.append(m.log_prob(action).item())
        self.states.append(state)
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

    def get_experience(self, final_obs=None, done=True):
        state = torch.from_numpy(final_obs).float()
        _, state_value = self.policy(state)
        final_value = state_value.detach() if not done else 0.0
        
        # rewards
        returns = self.discount_rewards(self.rewards, \
            self.dones, self.gamma, final_value)
        
        states = torch.stack(self.states).float()
    #     print(states.shape)
        old_actions = self.actions
    #     print(old_actions)
        old_logprobs = torch.tensor(self.logprobs).float()
    #     print(old_logprobs.shape)
        # print('collected experience')
        return states, torch.tensor(old_actions), old_logprobs, returns

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
        self.clip_val = 0.1
        self.c2 = 0.0001
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)
        self.n_epochs = 3

    def train(self, states, old_actions, old_logprobs, returns):
        # pass
        # print('processing experience')
        # return
         # PPO OLD VALUES
        for i in range(self.n_epochs):
            # Calculate needed values    
            p, v = policy.forward(states)
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
        # print('trained_on_experience')
        dist.barrier()

# N_EPS = 1500

# rewards_PPO = learn_PPO_single_threaded(N_EPS, 500)
# plt.plot(moving_average(rewards_PPO, 150), label='RMSprop')


def run(shared_policy, policy, rank, size):
    N_eps = 500
    ep_steps = 1500
    group = dist.new_group([i for i in range(size)])
    batch_update_freq = 30
    if rank != 0:
        if rank == 1: rewards = []
        env = gym.make('CartPole-v0')
        env.seed(rank) ; torch.manual_seed(rank) # seed everything
        env._max_episode_steps = ep_steps
        T = 0
        agent = PPO_Agent(shared_policy)

        for i_episode in range(N_eps):
            observation = env.reset()
            if rank == 1: total_r = 0
            for t in range(100000):
                T += 1
                action = agent.select_action(observation)
                observation, reward, done, info = env.step(action)

                agent.rewards.append(reward)
                agent.dones.append(done)

                if rank == 1: total_r += reward
                if T % batch_update_freq == 0:
                    # pass experience to the centralized trainer:
                    # states, old_actions, old_logprobs, returns = agent.get_experience(observation, done)
                    a,b,c,d = agent.get_experience(observation, done)
                    # print(f"""
                    #     a: {a}, shape: {a.shape}, {a.dtype} # torch.float32, 30,4
                    #     b: {b}, shape: {b.shape}, {b.dtype} # torch.int64, 30
                    #     c: {c}, shape: {c.shape}, {c.dtype} # torch.float32, 30
                    #     d: {d}, shape: {d.shape}, {d.dtype} # torch.float32, 30
                    # """)
                    # assert a.shape[0] == batch_update_freq, "qqqq"
                    # assert b.shape[0] == batch_update_freq, "zzzzz"
                    # assert c.shape[0] == batch_update_freq, 'ggggg'
                    # assert d.shape[0] == batch_update_freq, "ooooooo"
                    # print(a.shape, b.shape, c.shape, d.shape)
                    
                    dist.gather(a, gather_list=[], dst=0, group=group)
                    # dist.barrier()
                    # dist.gather(b, gather_list=[], dst=0, group=group)
                    # # dist.barrier()
                    # dist.gather(c, gather_list=[], dst=0, group=group)
                    # # # dist.barrier()
                    # dist.gather(d, gather_list=[], dst=0, group=group)
                    
                    agent.clear_experience()
                    # print('111')
                if done:
    #                 train_on_batch(0.99, observation, done)
                    if (i_episode + 1) % 100 == 0:                
                        if rank == 1: print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                    break
            if rank == 1: rewards.append(total_r)

        env.close()
    else:
        trainer = PPO_Centralized_Trainer(shared_policy, policy)
        old_states = [torch.zeros((30, 4), dtype=torch.float32) for i in range(size)]
        old_actions = [torch.zeros((30), dtype=torch.int64) for i in range(size)]
        old_logprobs = [torch.zeros((30), dtype=torch.float32) for i in range(size)]
        old_returns = [torch.zeros((30), dtype=torch.float32) for i in range(size)]

        while(True):
            dist.gather(old_states[0], gather_list=old_states, dst=0, group=group)
            # dist.gather(old_actions[0], gather_list=old_actions, dst=0, group=group)
            # dist.gather(old_logprobs[0], gather_list=old_logprobs, dst=0, group=group)
            # dist.gather(old_returns[0], gather_list=old_returns, dst=0, group=group)
            # print('----------')
            states = torch.cat(old_states[1:])
            # # print(states.shape)
            # actions = torch.cat(old_actions[1:])
            # # print(actions.shape)
            # logprobs = torch.cat(old_logprobs[1:])
            # # print(logprobs.shape)
            # returns = torch.cat(old_returns[1:])
            # print(returns.shape)
            # trainer.train(states, actions, logprobs, returns)
            # pass
            dist.barrier()


def init_process(shared_policy, policy, rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '30001'
    # os.environ['OMP_NUM_THREADS'] = '1'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(shared_policy, policy, rank, size)

if __name__ == '__main__':
    num_agents = 1
    # one thread 0 is reserved for centralized learner
    shared_policy = Policy()
    shared_policy.share_memory()
    policy = Policy()
    policy.load_state_dict(shared_policy.state_dict())

    size = num_agents + 1
    processes = []
    for rank in range(size):
        p = mp.Process(target=init_process, args=(shared_policy, policy, rank, size, run))
        p.start()
        processes.append(p)

    # processes = []
    # for rank in range(size):
    #     p = mp.Process(target=run, args=(shared_policy, policy, rank, size))
    #     p.start() ; processes.append(p)
    # for p in processes: p.join()

    for p in processes:
        p.join()


