import gym 
import torch

from pg_models import TD3
from actors import MLP

env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_scale = torch.ones(action_dim) * 2
device = "cuda" if torch.cuda.is_available() else "cpu"

episode_n = 200
trajectory_len = 200
buffer_size = 10000
batch_size = 64
tau = 1e-2
noise_mu = 0.0
noise_sigma = 0.2
noise_theta = 0.1
noise_decay = 0.95
noise_sigma_min = 0.

pi_lr = 1e-4
q_lr = 1e-3


pi_model = MLP(state_dim, action_dim, [400, 300], dropout=False, act_last=torch.tanh)
q_model = MLP(state_dim + action_dim, 1, [400, 300], dropout=False)

pi_optimizer = torch.optim.Adam(pi_model.parameters(), lr=pi_lr)
q_optimizer = torch.optim.Adam(q_model.parameters(), lr=q_lr)


agent = TD3(pi_model, q_model, pi_optimizer, q_optimizer,
             buffer_size, action_scale, batch_size,
             noise_mu=noise_mu, noise_sigma=noise_sigma, noise_theta=noise_theta, 
             noise_decay=noise_decay, noise_sigma_min=noise_sigma_min,              
             tau=tau, device=device)


for episode in range(episode_n):
    state = env.reset()
    total_reward = 0
    for t in range(trajectory_len):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward.item()
        agent.fit(state, action, reward, done, next_state)

        if done:
            break
        state = next_state
    print(f"{episode= } \t{total_reward= } ")
