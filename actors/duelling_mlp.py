from typing import List

from torch import nn

from actors import MLP


class DuellingMLP(MLP):
    def __init__(self, state_dim: int, action_n: int, sizes=List[int]):
        super(DuellingMLP, self).__init__(state_dim, sizes[-1], sizes, act_last=True)    
        self.action_n = action_n

        self.fc_adv = nn.Linear(sizes[-1], action_n)
        self.fc_value = nn.Linear(sizes[-1], 1)
        self.init_weights()
    
    def forward(self, x):
        hidden = super(DuellingMLP, self).forward(x)
        value = self.fc_value(hidden)
        adv = self.fc_adv(hidden)
        qvalues = value + adv - adv.mean(dim=1, keepdim=True)
        return qvalues
