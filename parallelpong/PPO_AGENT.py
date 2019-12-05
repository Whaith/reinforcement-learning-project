import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.FloatTensor)


def image_to_grey(obs, target_reso=(80, 80)):
    # print('here lol')
    return (np.dot(cv2.resize(obs[...,:3], dsize=target_reso), \
        [0.2989, 0.5870, 0.1140]).astype('float32')/255.0 + 0.15).round()

# source https://github.com/greydanus/baby-a3c/blob/master/baby-a3c.py
class NNPolicy(nn.Module): # an actor-critic neural network
    def __init__(self, channels, memsize, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.hidden = nn.Linear(288, memsize)
        self.critic_linear, self.actor_linear = nn.Linear(256, 1), nn.Linear(256, num_actions)

    def forward(self, x, train=True, hard=False):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = self.hidden(x.view(-1, 288))
        return F.softmax(self.actor_linear(x), -1), self.critic_linear(x)

class PPO_Agent():
    def __init__(self, policy, n_episodes):
        self.train_device = "cpu"
        self.policy = policy.to(self.train_device)
        self.eps = np.finfo(np.float32).eps.item()

        self.n_episodes_collect = n_episodes

        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.states = []
        self.hiddens = []
    
    def get_name(self):
        return "Ping Ping Pong"
        
    def select_action(self, state, save_values=True):
        # print(len(state))
        probs, _ = self.policy.forward((state))
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
            R = r + R*self.gamma
            returns.insert(0, R)
        returns = torch.tensor(returns)
        return returns
        # return (returns - returns.mean())/(returns.std() + self.eps)

    def get_experience(self, final_obs=None, done=True):
        _, state_value, _ = self.policy(state)
        final_value = state_value.detach() if not done else 0.0

        # rewards
        returns = self.discount_rewards(self.rewards, \
            self.dones, self.gamma, final_value)
        
        states = torch.stack(self.states).float()
        old_actions = torch.tensor(self.actions)
        old_logprobs = torch.tensor(self.logprobs).float()
        # print('collected experience')
        return states, torch.tensor(old_actions), old_logprobs, returns

    def clear_experience(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]