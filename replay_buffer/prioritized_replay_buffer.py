import random
from typing import List

import numpy as np
import torch

from replay_buffer.utils import SumTree
from replay_buffer import ReplayBuffer

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, state_size: int, action_size: int = 1, buffer_size: int = 10000, multi_step: int = 1, gamma: float = 1, 
                 eps: float = 1e-2, alpha: float = 0.1, beta: float = 0.1, device: str = 'cpu'):
        super(PrioritizedReplayBuffer, self).__init__(state_size, action_size, buffer_size, multi_step, gamma, device)

        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps
        
        self.tree = SumTree(self.size)
        self.real_size = 0
    
    def add(self, state: np.ndarray, action: float, reward: float, done: int, next_state: np.ndarray):
        self.tree.add(self.max_priority, self.count)

        super(PrioritizedReplayBuffer, self).add(state, action, reward, done, next_state)
    
    def sample(self, batch_size: int):
        batch = super(PrioritizedReplayBuffer, self).sample(batch_size)

        
        probs = self.priorities / self.tree.total
        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()

        return batch, weights, self.tree_idxs

    def update_priorities(self, data_idxs: torch.Tensor, priorities: torch.Tensor):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()
        for data_idx, priority in zip(data_idxs, priorities):
            priority = (np.abs(priority) + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def sample_idxs(self, batch_size: int) -> List[int]:
        tree_idxs, priorities, sample_idxs = map(torch.tensor, zip(*self.tree.stratified_sample(batch_size)))
        invalid_indices = [i % self.size for i in range(self.count - self.multi_step + 1, self.count)]
        
        for i in range(len(sample_idxs)):
            if sample_idxs[i] in invalid_indices:
                sample_idx = sample_idxs[i]
                while sample_idx in invalid_indices:
                    tree_idx, priority, sample_idx = self.tree.get()
                tree_idxs[i] = tree_idx
                priorities[i] = priority
                sample_idxs[i] = sample_idx
        self.tree_idxs = tree_idxs
        self.priorities = priorities
        return sample_idxs.long()
