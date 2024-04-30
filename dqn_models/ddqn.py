from typing import Union, Literal
from copy import deepcopy

import torch
from torch import nn
import numpy as np

from dqn_models import DQN


class DDQN(DQN):
    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            buffer_size: int = 10000,
            batch_size: int = 64, 
            gamma: int = 0.99, 
            multi_step: int = 1,
            prioritized: bool = False,
            epsilon: float = 1,
            device: str = 'cpu',
            update_interval: int = 1000,
            update_type=Literal['hard', 'soft'],
            alpha: float = 1
        ):
        super(DDQN, self).__init__(model, optimizer, buffer_size, batch_size, gamma, 
                                   multi_step, prioritized, epsilon, device)
        self.target_model = deepcopy(self.model)
        self.update_interval = update_interval
        self.update_counter = 0
        self.update_type = update_type
        self.alpha = alpha

    def get_next_state_values(self, next_states: torch.FloatTensor):
        next_qvalues = self.model(next_states).detach()
        
        next_actions = next_qvalues.max(dim=1)[1]
        next_qvalues_target = self.target_model(next_states)
        next_state_values = next_qvalues_target[range(len(next_actions)), next_actions]

        return next_state_values

    def fit(self, 
            state: np.ndarray, 
            action: np.ndarray, 
            reward: float, 
            done: bool, 
            next_state: np.ndarray,
            weights: Union[np.ndarray, None] = None
        ):
        self.update_counter += 1
        super(DDQN, self).fit(state, action, reward, done, next_state, weights)
        self.update_target()

    def update_target(self):
        if self.update_counter % self.update_interval == 0:
            if self.update_type == 'soft':
                for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                    target_param.data.copy_((1 - self.alpha) * target_param.data + self.alpha * param.data)
            else:
                self.target_model.load_state_dict(self.model.state_dict())
                

    