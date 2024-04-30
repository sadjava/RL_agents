from copy import deepcopy

import torch
from torch import nn
import numpy as np

from pg_models.utils import OUNoise
from replay_buffer import ReplayBuffer


class TD3:
    def __init__(self,
                 pi_model: nn.Module, 
                 q_model: nn.Module, 
                 pi_optimizer: torch.optim.Optimizer,
                 q_optimimizer: torch.optim.Optimizer,
                 buffer_size: int = 10000,
                 action_scale: torch.Tensor = None,
                 target_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_decay: int = 2,
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
        self.q1_model = q_model.to(self.device)
        self.q2_model = deepcopy(self.q1_model)

        self.pi_target_model = deepcopy(self.pi_model)
        self.q1_target_model = deepcopy(self.q1_model)
        self.q2_target_model = deepcopy(self.q2_model)

        self.action_dim = self.pi_model.action_n
        self.state_dim = self.pi_model.state_dim

        self.pi_optimizer = pi_optimizer
        self.q1_optimizer = q_optimimizer
        self.q2_optimizer = deepcopy(self.q1_optimizer)

        if action_scale is not None:
            self.action_scale = action_scale
        else:
            self.action_scale = torch.ones(self.action_dim)

        self.gamma = gamma
        self.tau = tau
        self.multi_step = multi_step
        self.noise_clip = noise_clip
        self.target_noise = target_noise
        self.policy_decay = policy_decay

        self.noise = OUNoise(self.action_dim, noise_mu, noise_theta, 
                             noise_sigma, noise_sigma_min, sigma_decay=noise_decay)
        
        self.exp_replay = ReplayBuffer(self.state_dim, self.action_dim, buffer_size=buffer_size, multi_step=multi_step, 
                                       gamma=gamma, device=device) 
        self.batch_size = batch_size
        self.updated_q = 0
    
    def get_action(self, state: np.ndarray) -> np.ndarray:
        pred_action = self.pi_model(torch.FloatTensor(state[None]).to(self.device)).cpu().detach().numpy()[0]
        action = self.action_scale * (pred_action + self.noise.sample())
        return np.clip(action, -self.action_scale, self.action_scale)
    
    def get_target_actions(self, states: torch.Tensor) -> torch.Tensor:
        pred_actions = self.pi_target_model(states) 
        epsilon = torch.randn_like(pred_actions) * self.target_noise
        actions = self.action_scale.to(self.device) * pred_actions + torch.clamp(epsilon, -self.noise_clip, self.noise_clip).to(self.device)
        return actions
    
    def update_target_model(self, model, target_model):
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
            self.updated_q += 1
            states, actions, rewards, dones, next_states = self.exp_replay.sample(self.batch_size)
            
            rewards = rewards.reshape(self.batch_size, 1)
            dones = dones.reshape(self.batch_size, 1)

            next_actions = self.get_target_actions(next_states)
            next_states_and_actions = torch.cat((next_states, next_actions), dim=1)

            next_q1_values = self.q1_target_model(next_states_and_actions)
            next_q2_values = self.q2_target_model(next_states_and_actions)

            targets = rewards + self.gamma * (1 - dones) * torch.min(next_q1_values, next_q2_values)

            states_and_actions = torch.cat((states, actions), dim=1)
            q1_values = self.q1_model(states_and_actions)
            q1_loss = torch.mean((q1_values - targets.detach()) ** 2)
            q1_loss.backward()
            self.q1_optimizer.step()
            self.q1_optimizer.zero_grad()
            
            q2_values = self.q2_model(states_and_actions)
            q2_loss = torch.mean((q2_values - targets.detach()) ** 2)
            q2_loss.backward()
            self.q2_optimizer.step()
            self.q2_optimizer.zero_grad()

            if self.updated_q % self.policy_decay == 0:
                pred_actions = self.pi_model(states)
                states_and_pred_actions = torch.cat((states, pred_actions), dim=1)
                pi_loss = -torch.mean(self.q1_model(states_and_pred_actions))
                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()
                self.update_target_model(self.q1_model, self.q1_target_model)
                self.update_target_model(self.q2_model, self.q2_target_model)
                self.update_target_model(self.pi_model, self.pi_target_model)

            self.noise.step()