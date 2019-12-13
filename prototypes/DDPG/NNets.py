import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def init_weights(model):
    for m in model.modules():
        if type(m) is torch.nn.Linear:
            torch.nn.init.normal_(m.weight)
            torch.nn.init.zeros_(m.bias)

class Policy_net(nn.Module):
    def __init__(self, ob_sp, act_sp):
        super(Policy_net, self).__init__()
        self.affine1 = nn.Linear(ob_sp, 200)
        self.affine2 = nn.Linear(200, 100)
        self.mean_head = nn.Linear(100, act_sp)
        init_weights(self)

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
        init_weights(self)

    def forward(self, state):
        x, action  = state
        x, action = x.to(device), action.to(device)
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))
        x = F.relu(self.value_head(x))
        x = torch.cat([x, action], 1)
        return self.value_head2(x)
    