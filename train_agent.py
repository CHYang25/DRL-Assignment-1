import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import torch

from dqn_agent import DQNAgent
from dynamic_taxi_env import DynamicTaxiEnv

from collections import deque
import click

class TrainingManager:

    def __init__(self, episodes):
        self.rewards_per_episode = []
        self.episode_len = []
        self.episode_cnt = 0
        self.episodes = episodes

    def add_episode(self, total_reward, episode_step):
        self.rewards_per_episode.append(total_reward)
        self.episode_len.append(episode_step)
        self.episode_cnt += 1

    def print_status(self, epsilon, last_n_episodes=100):
        avg_reward = np.mean(self.rewards_per_episode[-last_n_episodes:])
        print(f"🚀 Episode {self.episode_cnt + 1}/{self.episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

class ReplayBuffer:
    
    def __init__(self, buffer_size = 10000):
        self.data = {
            'obs': deque([], maxlen=buffer_size),
            'action': deque([], maxlen=buffer_size),
            'reward': deque([], maxlen=buffer_size),
            'next_obs': deque([], maxlen=buffer_size),
            'done': deque([], maxlen=buffer_size)
        }

    def add_transition(self, obs, action, reward, next_obs, done):
        self.data['obs'].append(obs)
        self.data['action'].append([action])
        self.data['reward'].append([reward])
        self.data['next_obs'].append(next_obs)
        self.data['done'].append([done])
        
    def __len__(self):
        return len(self.data['obs'])

    def __getitem__(self, idx):
        return {key: value[idx] for key, value in self.data.items()}
    
    def sample(self, batch_size):
        indices = np.random.permutation(self.__len__())[:batch_size]
        batch = {
            'obs': torch.stack([self.data['obs'][idx] for idx in indices]),
            'action': torch.stack([torch.tensor(self.data['action'][idx]) for idx in indices]),
            'reward': torch.stack([torch.tensor(self.data['reward'][idx]) for idx in indices]),
            'next_obs': torch.stack([self.data['next_obs'][idx] for idx in indices]),
            'done': torch.stack([torch.tensor(self.data['done'][idx]) for idx in indices])
        }
        return batch

def train_dqn(
        episodes=10, lr=1e-4, gamma=0.99, tau=0.005,
        epsilon_start=1.0, epsilon_end=0.01, 
        decay_rate=0.9999, batch_size=256
    ):
    
    # agent_x, agent_y, direction, key possession, door status, actions
    policy = DQNAgent(gamma=gamma, tau=tau, lr=lr, checkpoint=None)
    manager = TrainingManager(episodes=episodes)
    buffer = ReplayBuffer()

    epsilon = epsilon_start

    for episode in range(episodes):
        env = DynamicTaxiEnv(grid_size=random.choice(list(range(5, 11))))
        obs, _ = env.reset()
        policy.reset()

        observation = policy.get_observation(obs)  # Initially, the door is closed.

        done = False
        total_reward = 0
        episode_step = 0

        while not done:
            # TODO: Implement ε-greedy policy for action selection.
            action = policy.predict_action(observation, epsilon)

            # Execute the selected action.
            obs, reward, done, _ = env.step(action)
            next_observation = policy.get_observation(obs)
            episode_step += 1

            shaped_reward = policy.shaped_reward(observation, next_observation)
            reward += shaped_reward
            total_reward += reward

            # add transition to replay buffer
            buffer.add_transition(observation, action, reward, next_observation, done)

            # TODO: Apply Q-learning update rule (Bellman equation).
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                policy.update(batch)

            # Move to the next state.
            observation = next_observation

        manager.add_episode(total_reward=total_reward, episode_step=episode_step)
        epsilon = max(epsilon_end, epsilon * decay_rate)

        if (episode + 1) % 10 == 0:
            manager.print_status(epsilon=epsilon)

    policy.save_checkpoint()
    
    plt.plot(manager.rewards_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Taxi DQN Training Progress")
    plt.savefig("training_reward.png")

if __name__ == '__main__':
    train_dqn()