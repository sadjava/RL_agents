import random
from typing import Tuple, List
import math

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_size: int, action_size: int = 1, buffer_size: int = 10000, multi_step: int = 1, gamma: float = 1, device: str = 'cpu'):
        
        self.state = torch.empty((buffer_size, state_size), dtype=torch.float)
        self.action = torch.empty((buffer_size, action_size), dtype=torch.float)
        self.reward = torch.empty((buffer_size), dtype=torch.float)
        self.next_state = torch.empty((buffer_size, state_size), dtype=torch.float)
        self.done = torch.empty((buffer_size), dtype=torch.int)

        self.state_size = state_size
        self.size = buffer_size
        self.count = 0
        self.device = device
        self.real_size = 0
        self.multi_step = multi_step
        self.gamma = gamma

        self.cumulative_discount = torch.tensor([math.pow(gamma, i) for i in range(self.multi_step)], dtype=torch.float)

    def __len__(self) -> int:
        return self.real_size
    
    def add(self, state: np.ndarray, action: float, reward: float, done: int, next_state: np.ndarray):
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample_idxs(self, batch_size: int) -> List[int]:
        if self.real_size < self.size:
            idxs = random.sample(
                range(self.count - self.multi_step + 1),
                batch_size
            )
        else:
            shifted_idxs = random.sample(
                range(self.count, self.count + self.size - self.multi_step + 1),
                batch_size
            )
            idxs = [index % self.size for index in shifted_idxs]
        return idxs

    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.real_size - self.multi_step + 1 >= batch_size, "buffer contains less samples than batch size"
        
        
        sample_idxs = self.sample_idxs(batch_size)
        multi_step_returns = torch.empty((batch_size), dtype=torch.float)
        next_states = torch.empty((batch_size, self.state_size), dtype=torch.float)
        dones = torch.empty((batch_size), dtype=torch.int)

        for batch_idx, index in enumerate(sample_idxs):
            trajectory_idxs = [(index + i) % self.size for i in range(self.multi_step)]
            trajectory_dones = [self.done[i] for i in trajectory_idxs]

            done = any(trajectory_dones)
            if done:
                trajectory_len = trajectory_dones.index(True) + 1
                trajectory_idxs = trajectory_idxs[:trajectory_len]
            else:
                trajectory_len = self.multi_step
            
            multi_step_returns[batch_idx] = sum([self.reward[trajectory_idxs[i]] * self.cumulative_discount[i] 
                                                 for i in range(trajectory_len)])
            
            next_states[batch_idx] = self.next_state[(index + trajectory_len - 1) % self.size]
            dones[batch_idx] = done

        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].to(self.device),
            multi_step_returns.to(self.device),
            dones.to(self.device),
            next_states.to(self.device)
        )
        return batch