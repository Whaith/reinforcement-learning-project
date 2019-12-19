import os
import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import random
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributions import Categorical
import cv2
import glob
import wimblepong
import argparse
import time
from prototypes.utils import LinearSchedule
from tensorboardX import SummaryWriter

import sys
sys.path.append('.')


def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env', default='PingFuckingPong', type=str, help='gym environment')
    parser.add_argument('--processes', default=20, type=int, help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool, help='renders the atari environment')
    parser.add_argument('--test', default=False, type=bool, help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=20, type=int, help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int, help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float, help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float, help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float, help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int, help='hidden size of GRU')
    return parser.parse_args()
T_HORIZON = 64


ADAM_LR = 2.5e-5  # TODO ANNEAL this over the training to 0
# ADAM_LR = 0.0
N_EPOCHS = 3
DF = 0.99
N_ACTORS = 16
CLIP_PARAM = 0.1 # TODO ANNEAL this over during training to 0
ENTROPY_C2 = 0.01
MEM_SIZE = 256
IMAGE_H_W = 40

# V2 hyperparams
BATCH_SIZE = 64

def image_to_grey(obs, target_reso=(40, 40)):
    # print('here lol')
    return np.dot(cv2.resize(obs[...,:3], dsize=target_reso), \
        [0.2989, 0.5870, 0.1140]).astype('float32')/255.0

# source https://github.com/greydanus/baby-a3c/blob/master/baby-a3c.py
class NNPolicy(nn.Module): # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.linear = nn.Linear(288+288, 256)
        self.critic_linear, self.actor_linear = nn.Linear(256, 1), nn.Linear(256, num_actions)

    def forward(self, inputs, train=True, hard=False):
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.linear(x.view(-1, 288))
        return F.softmax(self.actor_linear(x), -1), self.critic_linear(x)

    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step

class PPO_Agent():
    def __init__(self, policy):
        self.gamma = DF
        self.batch_update_freq = BATCH_SIZE
        
        self.policy = policy
        self.repeat_action = None

        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.states = []
    
    def get_name(self):
        return "Ping Ping Ong"
        
    def select_action(self, state, save_values=True):
        # print(len(state))
        # state = state
        # state = torch.from_numpy(state).view(1, 1, IMAGE_H_W, IMAGE_H_W)
        probs, _ = self.policy.forward(state)
        m = Categorical(probs)
        action = m.sample()
        if save_values:
            self.actions.append(action.item())
            self.logprobs.append(m.log_prob(action).item())
            self.states.append(state)
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
        return returns

    def get_experience(self, final_obs=None, done=True):
        # state = torch.from_numpy(state).view(1, 1, IMAGE_H_W, IMAGE_H_W)
        _, state_value = self.policy(final_obs)
        final_value = state_value.detach() if not done else 0.0

        # rewards
        returns = self.discount_rewards(self.rewards, self.dones, self.gamma,
                                        final_value)
        
        states = torch.stack(self.states).float().view(-1, 3, IMAGE_H_W, IMAGE_H_W)
        old_actions = self.actions
        old_logprobs = torch.tensor(self.logprobs).float()
        # hiddens = torch.stack(self.hiddens).float()
        # hxs = torch.stack(self.hiddens).view(-1, MEM_SIZE).float()
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
        self.clip_val = CLIP_PARAM
        self.c2 = ENTROPY_C2
        self.optimizer = optim.Adam(self.policy.parameters(), lr=ADAM_LR)
        self.n_epochs = N_EPOCHS
        self.eps = np.finfo(np.float32).eps.item()

    def train(self, states, old_actions, old_logprobs, returns, alpha=None):
        
        # print('processing experience')
        for i in range(self.n_epochs):
            # Calculate needed values    
            # print('qqqqq', states.shape, hiddens.shape)
            p, v = policy.forward(states)
            m = Categorical(p)
            c = m.log_prob(old_actions)
            entr = m.entropy()

            # value fn loss
            loss_vf = F.mse_loss(v.squeeze(-1), returns.squeeze(-1))

            # anneal the clip valueh
            if alpha is not None:
                self.clip_val = CLIP_PARAM*alpha

            # surrogate loss
            advantage = returns - v.detach()
            advantage = (advantage - advantage.mean()) / \
                        (advantage.std() + self.eps)

            r_ts = torch.exp(c - old_logprobs)
            loss_surr = - (torch.min(r_ts * advantage,
                           torch.clamp(r_ts, 1-self.clip_val, 1+self.clip_val)
                           * advantage)).mean()
            
            # maximize entropy bonus
            loss_entropy = - self.c2 * entr.mean()
            
            # the total_loss
            loss_total = loss_vf + loss_surr + loss_entropy
            if alpha != None:
                for g in self.optimizer.param_groups:
                    g['lr'] = ADAM_LR*alpha
            # step
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()

        self.shared_policy.load_state_dict(self.policy.state_dict())

        dist.barrier()

def printlog(args, s, end='\n', mode='a'):
    print(s, end=end) ; f=open(args.save_dir+'log.txt',mode) ; f.write(s+'\n') ; f.close()

def run(shared_policy, policy, rank, size, info, args):
    # print(info['frames'])
    N_eps = int(1e6)
    group = dist.new_group([i for i in range(size)])
    batch_update_freq = T_HORIZON
    # Agent getting the training data

    if rank != 0:
        env = gym.make("WimblepongVisualSimpleAI-v0")
        action_space = env.action_space.n
        env.seed(rank) ; torch.manual_seed(rank) # seed everything
        if rank == 1: writer = SummaryWriter()
        T = 0
        agent = PPO_Agent(shared_policy)
        start_time = last_disp_time = time.time()
        for i_episode in range(N_eps):
            init_obs = image_to_grey(env.reset())
            obs_deq = deque([torch.from_numpy(init_obs) for i in range(3)], maxlen=3)
            if rank == 1: total_r = 0
            done = True
            # hx = torch.zeros(1, MEM_SIZE) if done else hx.detach()
            epr = 0
            while True:
                T += 1
                # print('INFO',info)
                info['frames'].add_(1)
                num_frames = int(info['frames'].item())
                # num_frames = 0
                # print('-obssss shapee', observation.shape)
                # print(obs_deq)
                action = agent.select_action(torch.stack(list(obs_deq)).view(-1, 3, 40, 40))
                observation, reward, done, _ = env.step(action)
                obs_deq.append(torch.from_numpy(image_to_grey(observation)))
                # check that the reward is always non zero
                if np.abs(reward) < 5:
                    reward = 0.001
                agent.rewards.append(reward)
                epr += reward
                agent.dones.append(done)
                # print('aaa')
                if rank == 1: total_r += reward
                if T % batch_update_freq == 0:
                    # print('list', obs_deq)
                    # print('zzzzz')

                    # print(torch.stack(list(obs_deq)).shape)
                    next_obs = torch.stack(list(obs_deq)).view(-1, 3, 40, 40)
                    a,b,c,d = agent.get_experience(next_obs, done)
                    # print("GETTING EXP")
                    # print(a.size(), a.dtype)
                    # print(h.size(), h.dtype)
                    # print(f"""
                    # a: {a[0]}, {a.size()} \n
                    # b: {b}, {b.size()} \n
                    # c: {c}, {c.size()} \n
                    # h: {d}, {d.size()} \n
                    # d: {h[0]}, {h.size()} \n
                    # """)
                    dist.gather(a, gather_list=[], dst=0, group=group)
                    # dist.barrier()
                    dist.gather(b, gather_list=[], dst=0, group=group)
                    # dist.barrier()
                    dist.gather(c, gather_list=[], dst=0, group=group)
                    # # dist.barrier()
                    dist.gather(d, gather_list=[], dst=0, group=group)
                    # print('HIDDENS shape: ', h, h.shape)
                    # dist.gather(h, gather_list=[], dst=0, group=group)
                    
                    agent.clear_experience()

                if num_frames % int(2e6) == 0:
                    printlog(args, '\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
                    torch.save(shared_policy.state_dict(), args.save_dir+'model.{:.0f}.tar'.format(num_frames/1e6))

                if rank == 1 and time.time() - last_disp_time > 60: # print info ~ every minute
                    if rank == 1: writer.add_scalar('running reward', info['run_epr'].item(), num_frames)
                    elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                    printlog(args, 'time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}'
                        .format(elapsed, info['episodes'].item(), num_frames/1e6,
                        info['run_epr'].item()))
                    last_disp_time = time.time()


                if done:
                    info['episodes'] += 1
                    interp = 1 if info['episodes'][0] == 1 else 1 - 0.99
                    info['run_epr'].mul_(1-interp).add_(interp * epr)
                    # print(f"rank: {rank}, episode: {i_episode}")
                    # if (i_episode + 1) % 100 == 0:                
                    #     if rank == 1: print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                    break

            # if rank == 1: rewards_deque.append(total_r)
        env.close()
    # centralized actor training on the training data
    else:
        trainer = PPO_Centralized_Trainer(shared_policy, policy)
        old_states = [torch.zeros((T_HORIZON, 3, IMAGE_H_W, IMAGE_H_W), dtype=torch.float32) for i in range(size)]
        old_actions = [torch.zeros((T_HORIZON), dtype=torch.int64) for i in range(size)]
        old_logprobs = [torch.zeros((T_HORIZON), dtype=torch.float32) for i in range(size)]
        old_returns = [torch.zeros((T_HORIZON), dtype=torch.float32) for i in range(size)]
        old_states = [torch.zeros((1, 3, IMAGE_H_W, IMAGE_H_W), dtype=torch.float32) for i in range(size)]
        # old_hiddens = [torch.zeros((T_HORIZON, MEM_SIZE), dtype=torch.float32) for i in range(size)]
        scheduler = LinearSchedule(60e6, final_p = 0.3, initial_p= 1.0)
        while(True):
            num_frames = int(info['frames'].item())
            dist.gather(old_states[0], gather_list=old_states, dst=0, group=group)
            dist.gather(old_actions[0], gather_list=old_actions, dst=0, group=group)
            dist.gather(old_logprobs[0], gather_list=old_logprobs, dst=0, group=group)
            dist.gather(old_returns[0], gather_list=old_returns, dst=0, group=group)
            # dist.gather(old_hiddens[0], gather_list=old_hiddens, dst=0, group=group)
            states = torch.cat(old_states[1:])
            actions = torch.cat(old_actions[1:])
            logprobs = torch.cat(old_logprobs[1:])
            returns = torch.cat(old_returns[1:])
            # hiddens = torch.cat(old_hiddens[1:])
            # print(states.shape, actions.shape, logprobs.shape, returns.shape, hiddens.shape)
            trainer.train(states, actions, logprobs, returns, scheduler.value(num_frames))

def init_process(shared_policy, policy, rank, size, fn, info, args, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '30029'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['GLOO_SOCKET_IFNAME'] = 'eno1'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(shared_policy, policy, rank, size, info, args)

def evaluate_model():
    pass

if __name__ == '__main__':
    args = get_args()

    num_agents = 2
    # one thread 0 is reserved for centralized learner
    env = gym.make("WimblepongVisualSimpleAI-v0")

    shared_policy = NNPolicy(3, MEM_SIZE, env.action_space.n)
    shared_policy.try_load(args.env)
    shared_policy.share_memory()
    policy = NNPolicy(3, MEM_SIZE, env.action_space.n)
    policy.load_state_dict(shared_policy.state_dict())
    
    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'episodes', 'frames']}

    args.save_dir = '{}/'.format(args.env.lower()) # keep the directory structure simple
    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else None # make dir to save models etc.
    size = num_agents + 1
    processes = []
    
    for rank in range(size):
        p = mp.Process(target=init_process, args=(shared_policy, policy, rank, size, run, info, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


