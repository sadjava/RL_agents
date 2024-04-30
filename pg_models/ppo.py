import torch 
from torch import nn
from torch.distributions import Normal
import numpy as np

class PPO(nn.Module):
    def __init__(self, 
                 pi_model: nn.Module, 
                 v_model: nn.Module,
                 pi_optimizer: torch.optim.Optimizer,
                 v_optimizer: torch.optim.Optimizer,
                 action_scale: float = 1.,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,
                 epoch_n: int = 30,
                 device: str = 'cpu'
                 ):
        super().__init__()

        self.batch_size = batch_size
        self.action_scale = action_scale
        self.gamma = gamma
        self.epsilon = epsilon
        self.epoch_n = epoch_n
        self.device = device

        self.pi_model = pi_model.to(self.device)
        self.v_model = v_model.to(self.device)
        self.pi_optimizer = pi_optimizer
        self.v_optimizer = v_optimizer

        self.action_dim = self.pi_model.action_n // 2

    def get_dist(self, states: torch.Tensor) -> torch.distributions.Distribution:
        mean, log_std = torch.split(self.pi_model(states), self.action_dim, dim=1)
        dist = Normal(mean, torch.exp(log_std))
        return dist

    def get_action(self, state: np.ndarray) -> np.ndarray:
        dist = self.get_dist(torch.FloatTensor(state[None]).to(self.device))
        action = self.action_scale * dist.sample()
        return action.cpu().numpy().reshape(self.action_dim)

    def fit(self, 
            states: np.ndarray,
            actions: np.ndarray,
            rewards: np.ndarray,
            dones: np.ndarray
        ):
        rewards, dones = rewards.reshape(-1, 1), dones.reshape(-1, 1)
        
        returns = np.zeros(rewards.shape)
        returns[-1] = rewards[-1]
        for t in range(states.shape[0] - 2, -1, -1):
            returns[t] = rewards[t] + (1 - dones[t]) *self.gamma * returns[t + 1]
        
        states, actions, returns, rewards = map(torch.FloatTensor, [states, actions, returns, rewards])
        states, actions, returns, rewards = states.to(self.device), actions.to(self.device), returns.to(self.device), rewards.to(self.device)
        
        dist = self.get_dist(states)
        old_log_probs = dist.log_prob(actions).detach()

        for epoch in range(self.epoch_n):
            idxs = np.random.permutation(states.shape[0])
            for i in range(0, states.shape[0], self.batch_size):
                idx = idxs[i:i + self.batch_size]
                batch_states, batch_actions, batch_returns, batch_old_log_probs = states[idx], actions[idx], returns[idx], old_log_probs[idx]

                batch_advantage = batch_returns.detach() - self.v_model(batch_states)

                batch_dist = self.get_dist(batch_states)
                batch_log_probs = batch_dist.log_prob(batch_actions)

                ratio = torch.exp(batch_log_probs - batch_old_log_probs)

                pi_loss1 = ratio * batch_advantage.detach()
                pi_loss2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantage.detach()

                pi_loss = -torch.mean(torch.min(pi_loss1, pi_loss2))
                pi_loss.backward()
                self.pi_optimizer.step()
                self.pi_optimizer.zero_grad()

                v_loss = torch.mean(batch_advantage ** 2)
                v_loss.backward()
                self.v_optimizer.step()
                self.v_optimizer.zero_grad()
                

                                                

