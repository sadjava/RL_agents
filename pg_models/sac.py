from copy import deepcopy

import torch
from torch import nn
from torch.distributions import Normal
import numpy as np

from replay_buffer import ReplayBuffer


class SAC:
    def __init__(self,
                 pi_model: nn.Module, 
                 q_model: nn.Module, 
                 pi_optimizer: torch.optim.Optimizer,
                 q_optimimizer: torch.optim.Optimizer,
                 buffer_size: int = 10000,
                 action_scale: torch.Tensor = None,
                 batch_size: int = 64,
                 multi_step: int = 1,
                 alpha: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 1e-2,
                 device: str = 'cpu'
        ):

        self.device = device 

        self.pi_model = pi_model.to(self.device)
        self.q1_model = q_model.to(self.device)
        self.q2_model = deepcopy(self.q1_model)

        self.q1_target_model = deepcopy(self.q1_model)
        self.q2_target_model = deepcopy(self.q2_model)

        self.action_dim = self.pi_model.action_n // 2
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
        self.alpha = alpha
        self.multi_step = multi_step
        
        self.exp_replay = ReplayBuffer(self.state_dim, self.action_dim, buffer_size=buffer_size, multi_step=multi_step, 
                                       gamma=gamma, device=device) 
        self.batch_size = batch_size
        self.updated_q = 0
    
    def get_action(self, state):
        state = torch.FloatTensor(state[None]).to(self.device)
        action, _ = self.predict_actions(state)
        return action.cpu().squeeze(1).detach().numpy()

    def predict_actions(self, states):
        means, log_stds = torch.split(self.pi_model(states), self.action_dim, dim=1)
        dists = Normal(means, torch.exp(log_stds))
        actions = dists.rsample()
        log_probs = dists.log_prob(actions)
        return actions, log_probs
    
    def update_model(self, loss, optimizer, model=None, target_model=None):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if model != None and target_model != None:
            for target_param, param in zip(target_model.parameters(), model.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )
    
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

            next_actions, next_log_probs = self.predict_actions(next_states)
            next_states_and_actions = torch.cat((next_states, next_actions), dim=1)

            next_q1_values = self.q1_target_model(next_states_and_actions)
            next_q2_values = self.q2_target_model(next_states_and_actions)

            next_min_q_values = torch.min(next_q1_values, next_q2_values)

            targets = rewards + self.gamma * (1 - dones) * (next_min_q_values - self.alpha * next_log_probs)

            states_and_actions = torch.cat((states, actions), dim=1)
            q1_values = self.q1_model(states_and_actions)
            q1_loss = torch.mean((q1_values - targets.detach()) ** 2)
            
            q2_values = self.q2_model(states_and_actions)
            q2_loss = torch.mean((q2_values - targets.detach()) ** 2)
            
            self.update_model(q1_loss, self.q1_optimizer, self.q1_model, self.q1_target_model)
            self.update_model(q2_loss, self.q2_optimizer, self.q2_model, self.q2_target_model)

            pred_actions, log_probs = self.predict_actions(states)
            states_and_pred_actions = torch.cat((states, pred_actions), dim=1)
            pred_q1_values = self.q1_model(states_and_pred_actions)
            pred_q2_values = self.q2_model(states_and_pred_actions)
            pred_min_q_values = torch.min(pred_q1_values, pred_q2_values)
            pi_loss = -torch.mean(pred_min_q_values - self.alpha * log_probs)
            self.update_model(pi_loss, self.pi_optimizer)
