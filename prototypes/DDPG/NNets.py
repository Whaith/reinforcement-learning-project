import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Policy_net(nn.Module):
    def __init__(self, ob_sp, act_sp):
        super(Policy_net, self).__init__()
        self.affine1 = nn.Linear(ob_sp, 200)
        self.affine2 = nn.Linear(200, 100)
        self.mean_head = nn.Linear(100, act_sp)
        # self.reset_parameters()
        # self.sigma = torch.nn.Parameter(torch.tensor([ self.sigma], requires_grad=True))
    # def reset_parameters(self):
    #     self.affine1.weight.data.uniform_(*hidden_init(self.affine1))
    #     self.affine2.weight.data.uniform_(*hidden_init(self.affine2))
    #     self.mean_head.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        # -2 to 2 with tanh
        mean = self.mean_head(x)
        return  mean

class Q_net(nn.Module):
    def __init__(self, ob_sp, act_sp):
        super(Q_net, self).__init__()
        self.act_sp = act_sp
        self.ob_sp = ob_sp
        self.affine1 = nn.Linear(self.ob_sp, 200)
        self.affine2 = nn.Linear(200, 100)
        self.value_head = nn.Linear(100, 10)
        self.value_head2 = nn.Linear(self.act_sp + 10, 1)
        # self.reset_parameters()
    # def reset_parameters(self):
    #     self.affine1.weight.data.uniform_(*hidden_init(self.affine1))
    #     self.affine2.weight.data.uniform_(*hidden_init(self.affine2))
    #     self.value_head.weight.data.uniform_(*hidden_init(self.value_head))
    #     self.value_head2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x, action  = state
        x, action = x.to(device), action.to(device)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.value_head(x))
        x = torch.cat([x, action], 1)
        return self.value_head2(x)
    