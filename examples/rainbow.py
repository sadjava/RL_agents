import gym
import torch

from dqn_models import DDQN
from actors import Rainbow


env = gym.make('LunarLander-v2')
state_dim = env.observation_space.shape[0]
action_n = env.action_space.n 

batch_size = 64
trajectory_n = 200
trajectory_len = 1000
buffer_size = 10000
gamma = 0.99
multi_step = 5
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
update_interval = 100
update_type = 'soft'
alpha = 0.2

model = Rainbow(state_dim, action_n, [128, 128])
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

agent = DDQN(model, optimizer, buffer_size=buffer_size, 
            batch_size=batch_size, prioritized=True,
            gamma=gamma, multi_step=multi_step, epsilon=1,
            update_interval=update_interval, update_type=update_type,
            alpha=alpha, device=device)


for i in range(trajectory_n): 
    total_reward = 0
    state = env.reset()
    for _ in range(trajectory_len): 
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)

        agent.fit(state, action, reward, done, next_state)

        state = next_state
        total_reward += reward

        # env.render()
        if done:
            break
    
    print(f"trajectory {i}\t{total_reward = }")

state = env.reset()
for _ in range(trajectory_len): 
    action = agent.get_action(state)
    state, reward, done, info = env.step(action)

    env.render()
    if done:
        break