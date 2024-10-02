import numpy as np
import torch
from collections import deque
import gymnasium as gym
from agent import Agent

# Initialize environment and agent
env = gym.make("LunarLander-v2")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size, action_size)

# Hyperparameters
number_episodes = 2000
maximum_number_timesteps_per_episode = 1000
epsilon_starting_value = 1.0
epsilon_ending_value = 0.01
epsilon_decay_value = 0.995
epsilon = epsilon_starting_value
scores_on_100_episodes = deque(maxlen=100)

# Training loop
for episode in range(1, number_episodes + 1):
    state, _ = env.reset()
    score = 0
    for t in range(maximum_number_timesteps_per_episode):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    scores_on_100_episodes.append(score)
    epsilon = max(epsilon_ending_value, epsilon_decay_value * epsilon)

    print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_on_100_episodes):.2f}', end="")
    if episode % 100 == 0:
        print(f'\rEpisode {episode}\tAverage Score: {np.mean(scores_on_100_episodes):.2f}')
    if np.mean(scores_on_100_episodes) >= 200.0:
        print(f'\nEnvironment solved in {episode-100} episodes!\tAverage Score: {np.mean(scores_on_100_episodes):.2f}')
        torch.save(agent.local_qnetwork.state_dict(), 'checkpoint.pth')
        break
