import gym 
import torch

from pg_models import SAC
from actors import MLP

env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_scale = torch.ones(action_dim) * 2
device = "cuda" if torch.cuda.is_available() else "cpu"

episode_n = 100
trajectory_len = 200
buffer_size = 10000
batch_size = 64
tau = 1e-2
alpha = 1e-3


pi_lr = 1e-3
q_lr = 1e-3


pi_model = MLP(state_dim, 2 * action_dim, [400, 300], dropout=False, act_last=torch.tanh)
q_model = MLP(state_dim + action_dim, 1, [400, 300], dropout=False)

pi_optimizer = torch.optim.Adam(pi_model.parameters(), lr=pi_lr)
q_optimizer = torch.optim.Adam(q_model.parameters(), lr=q_lr)


agent = SAC(pi_model, q_model, pi_optimizer, q_optimizer,
             buffer_size, action_scale, batch_size,
             alpha = alpha,
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

state = env.reset()
for _ in range(trajectory_len): 
    action = agent.get_action(state)
    next_state, reward, done, info = env.step(action)

    env.render()
    if done:
        break
    state = next_state