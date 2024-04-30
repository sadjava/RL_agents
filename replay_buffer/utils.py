from typing import Tuple
import random

import numpy as np


class SumTree:
    def __init__(self, size: int):
        self.nodes = [0] * (2 * size - 1)
        self.data = [0] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self) -> int:
        return self.nodes[0]
    
    def update(self, data_index: int, value: float):
        index = data_index + (self.size - 1)
        change = value - self.nodes[index]

        self.nodes[index] = value
        
        parent = (index - 1) // 2 
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2
    
    def add(self, value: float, data: float):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum: float = None) -> Tuple[int, float, float]:
        if cumsum is None:
            cumsum = random.random() * self.total
        assert cumsum <= self.total

        index = 0
        while 2 * index + 1 < len(self.nodes):
            left, right = 2 * index + 1, 2 * index + 2
            if cumsum <= self.nodes[left]:
                index = left
            else:
                index = right
                cumsum = cumsum - self.nodes[left]
                
        data_index = index - (self.size - 1)
        return data_index, self.nodes[index], self.data[data_index]

    def stratified_sample(self, batch_size):
        if self.total == 0.0:
            raise Exception('Cannot sample from an empty sum tree.')
        
        bounds = np.linspace(0., self.total, batch_size + 1)
        assert len(bounds) == batch_size + 1
        segments = [(bounds[i], bounds[i+1]) for i in range(batch_size)]
        query_values = [random.uniform(x[0], x[1]) for x in segments]
        return [self.get(x) for x in query_values]