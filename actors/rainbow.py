from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from actors import MLP, NoisyLinear


class Rainbow(MLP):
    def __init__(self, state_dim: int, action_n: int, sizes: List[int] = [128, 128], 
                 n_atoms: int = 51, vmin: float = -10, vmax: float = 10):
        super(Rainbow, self).__init__(state_dim, sizes[-1], sizes, act_last=torch.relu)    
        self.action_n = action_n
        self.n_atoms = n_atoms
        self.register_buffer('support', torch.linspace(vmin, vmax, n_atoms))

        self.fc_value = NoisyLinear(sizes[-1], n_atoms)
        self.fc_adv = NoisyLinear(sizes[-1], action_n * n_atoms)
        self.init_weights()
    
    def forward(self, x):
        distribution = self.distribution(x)
        qvalues = torch.sum(distribution * self.support, dim=2)
        return qvalues
    
    def logits(self, x):
        hidden = super(Rainbow, self).forward(x)
        value = self.fc_value(hidden)
        adv = self.fc_adv(hidden)

        value = value.view(-1, 1, self.n_atoms)
        adv = adv.view(-1, self.action_n, self.n_atoms)

        return value + adv - adv.mean(dim=1, keepdim=True)
    
    def distribution(self, x):
        logits = self.logits(x)
        return F.softmax(logits, dim=-1)
