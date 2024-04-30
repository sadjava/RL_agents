from typing import Tuple, Union

import torch 
from torch import nn
import numpy as np

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQN:
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
            device: str = 'cpu'
        ):
        self.action_n = model.action_n
        self.state_dim = model.state_dim

        self.batch_size = batch_size
        self.exp_replay = ReplayBuffer(self.state_dim, buffer_size=buffer_size, multi_step=multi_step, 
                                       gamma=gamma, device=device) if not prioritized \
            else PrioritizedReplayBuffer(self.state_dim, buffer_size=buffer_size, multi_step=multi_step, 
                                       gamma=gamma, device=device)
        

        self.multi_step = multi_step
        self.gamma = gamma
        self.epsilon = epsilon
        self.prioritized = prioritized
        
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        qvalues = self.model(torch.FloatTensor(state[None]).to(self.device)).detach().cpu().numpy()[0]
        prob = np.ones(self.action_n) * self.epsilon / self.action_n
        argmax_action = np.argmax(qvalues)
        prob[argmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.action_n), p=prob)
        return action

    def get_next_state_values(self, next_states: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        next_qvalues = self.model(next_states)

        next_state_values = next_qvalues.max(dim=1)[0]
        return next_state_values

    def fit(self, 
            state: np.ndarray, 
            action: np.ndarray, 
            reward: float, 
            done: bool, 
            next_state: np.ndarray,
            weights: Union[np.ndarray, None] = None
        ):
        self.exp_replay.add(state, action, reward, done, next_state)

        if len(self.exp_replay) - self.multi_step + 1 >= self.batch_size:
            if not self.prioritized:
                states, actions, rewards, dones, next_states = self.exp_replay.sample(self.batch_size)
            else:
                batch, weights, tree_idxs = self.exp_replay.sample(self.batch_size)
                states, actions, rewards, dones, next_states = batch

            actions = actions.reshape(-1).long()

            qvalues = self.model(states)
            qvalues_for_actions = qvalues[range(len(actions)), actions]

            next_state_values = self.get_next_state_values(next_states)
            
            target_qvalues_for_actions = rewards + (1 - dones) * self.gamma * next_state_values

            if weights is None:
                weights = torch.ones_like(qvalues_for_actions, device=self.device)
            else: 
                weights = weights.to(self.device)

            td_error = torch.abs(qvalues_for_actions - target_qvalues_for_actions).detach()

            if self.prioritized:
                self.exp_replay.update_priorities(tree_idxs, td_error)
                                                
            
            loss = torch.mean((target_qvalues_for_actions.detach() - qvalues_for_actions) ** 2 * weights)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.epsilon > 0.01:
                self.epsilon *= 0.995