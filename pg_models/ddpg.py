from typing import Union
from copy import deepcopy

import torch
from torch import nn
import numpy as np

from pg_models.utils import OUNoise
from replay_buffer import ReplayBuffer


class DDPG:
    def __init__(self,
                 pi_model: nn.Module, 
                 q_model: nn.Module, 
                 pi_optimizer: torch.optim.Optimizer,
                 q_optimimizer: torch.optim.Optimizer,
                 buffer_size: int = 10000,
                 action_scale: torch.Tensor = None,
                 batch_size: int = 64,
                 multi_step: int = 1,
                 gamma: float = 0.99,
                 tau: float = 1e-2,
                 noise_mu: float = 0.0,
                 noise_sigma: float = 0.5,
                 noise_theta: float = 0.1,
                 noise_decay: float = 0.99,
                 noise_sigma_min: float = 0.01,
                 device: str = 'cpu'
        ):

        self.device = device 

        self.pi_model = pi_model.to(self.device)
        self.q_model = q_model.to(self.device)

        self.pi_target_model = deepcopy(self.pi_model)
        self.q_target_model = deepcopy(self.q_model)

        self.action_dim = self.pi_model.action_n
        self.state_dim = self.pi_model.state_dim

        self.pi_optimizer = pi_optimizer
        self.q_optimizer = q_optimimizer

        if action_scale is not None:
            self.action_scale = action_scale
        else:
            self.action_scale = torch.ones(self.action_dim, device=self.device)

        self.gamma = gamma
        self.tau = tau
        self.multi_step = multi_step

        self.noise = OUNoise(self.action_dim, noise_mu, noise_theta, 
                             noise_sigma, noise_sigma_min, sigma_decay=noise_decay)
        
        self.exp_replay = ReplayBuffer(self.state_dim, self.action_dim, buffer_size=buffer_size, multi_step=multi_step, 
                                       gamma=gamma, device=device) 
        self.batch_size = batch_size
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        pred_action = self.pi_model(torch.FloatTensor(state[None]).to(self.device)).cpu().detach().numpy()[0]
        action = self.action_scale * (pred_action + self.noise.sample())
        return np.clip(action, -self.action_scale, self.action_scale)
    
    def update_target_model(self, model, target_model, optimizer, loss):
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        for param, taret_param in zip(model.parameters(), target_model.parameters()):
            taret_param.data.copy_(self.tau * param.data + (1 - self.tau) * taret_param.data)
    
    def fit(self, 
            state: np.ndarray, 
            action: np.ndarray, 
            reward: float, 
            done: bool, 
            next_state: np.ndarray
        ):
        self.exp_replay.add(state, action, reward, done, next_state)

        if len(self.exp_replay) - self.multi_step + 1 >= self.batch_size:
            states, actions, rewards, dones, next_states = self.exp_replay.sample(self.batch_size)
            
            rewards = rewards.reshape(self.batch_size, 1)
            dones = dones.reshape(self.batch_size, 1)

            next_actions = self.action_scale.to(self.device) * self.pi_target_model(next_states)
            next_states_and_actions = torch.cat((next_states, next_actions), dim=1)
            next_q_values = self.q_target_model(next_states_and_actions)

            targets = rewards + self.gamma * (1 - dones) * next_q_values

            states_and_actions = torch.cat((states, actions), dim=1)
            q_values = self.q_model(states_and_actions)
            q_loss = torch.mean((q_values - targets.detach()) ** 2)
            self.update_target_model(self.q_model, self.q_target_model, self.q_optimizer, q_loss)

            pred_actions = self.pi_model(states)
            states_and_pred_actions = torch.cat((states, pred_actions), dim=1)
            pi_loss = -torch.mean(self.q_model(states_and_pred_actions))
            self.update_target_model(self.pi_model, self.pi_target_model, self.pi_optimizer, pi_loss)

            self.noise.step()