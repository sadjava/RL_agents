import torch
import numpy as np
import gym

from actors import MLP
from pg_models import PPO


env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_scale = 2 # Pendulum action range [-2, 2]

pi_lr = 1e-4
v_lr = 1e-4
batch_size = 128
gamma = 0.9
epsilon = 0.2
epoch_n = 30
device = 'cuda' if torch.cuda.is_available() else 'cpu'

episode_n = 50
trajectory_n = 20
trajectory_len = 200

pi_model = MLP(state_dim, 2 * action_dim, [64], dropout=False, act_last=torch.tanh)
v_model = MLP(state_dim, 1, [64], dropout=False)

pi_optimizer = torch.optim.Adam(pi_model.parameters(), lr=pi_lr)
v_optimizer = torch.optim.Adam(v_model.parameters(), lr=v_lr)

agent = PPO(pi_model, v_model, pi_optimizer, v_optimizer, action_scale=action_scale,
            gamma=gamma, epsilon=epsilon, batch_size=batch_size,
            epoch_n=epoch_n,device=device)

for episode in range(episode_n):
    states, actions, rewards, dones = [], [], [], []
    total_rewards = []
    for _ in range(trajectory_n):
        total_reward = 0
        state = env.reset()
        for t in range(trajectory_len):
            states.append(state)
            
            action = agent.get_action(state)
            actions.append(action)
            
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            dones.append(done)

            total_reward += reward
        total_rewards.append(total_reward)
    print(f"episode {episode}\tmean_total_reward = {np.mean(total_rewards)}")
    states, actions, rewards, dones = map(np.array, [states, actions, rewards, dones])
    agent.fit(states, actions, rewards, dones)
        
state = env.reset()
for _ in range(trajectory_len): 
    action = agent.get_action(state)
    state, reward, done, info = env.step(action)

    env.render()
    if done:
        break