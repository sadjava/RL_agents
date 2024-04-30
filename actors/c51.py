from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from actors import MLP

class C51(MLP):
    def __init__(self, state_dim: int, action_n: int, sizes: List[int] = [128, 128], 
                 n_atoms: int = 51, vmin: float = -10, vmax: float = 10):
        super(C51, self).__init__(state_dim, sizes[-1], sizes, act_last=True)    
        self.action_n = action_n
        self.n_atoms = n_atoms
        self.register_buffer('support', torch.linspace(vmin, vmax, n_atoms))

        self.fc = nn.Linear(sizes[-1], action_n * n_atoms)
        self.init_weights()
    
    def forward(self, x):
        distribution = self.distribution(x)
        qvalues = torch.sum(distribution * self.support, dim=2)
        return qvalues
    
    def logits(self, x):
        hidden = super(C51, self).forward(x)
        x = self.fc(hidden)
        return x.view(-1, self.action_n, self.n_atoms)
    
    def distribution(self, x):
        logits = self.logits(x)
        return F.softmax(logits, dim=-1)