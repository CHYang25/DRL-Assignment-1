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

import wandb
from datetime import datetime

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
        print(f"🚀 Episode {self.episode_cnt}/{self.episodes}, Average Reward: {avg_reward:.2f}, Epsilon: {epsilon:.3f}")

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

@click.command()
@click.option('-n', '--num_episodes', default=10000)
@click.option('-l', '--lr', default=1e-4)
@click.option('-g', '--gamma', default=0.99)
@click.option('-t', '--tau', default=0.005)
@click.option('--epsilon_start', default=1.0)
@click.option('--epsilon_end', default=0.05)
@click.option('--decay_rate', default=0.9995)
@click.option('--batch_size', default=256)
@click.option('--buffer_size', default=50000)
def train_dqn(
        num_episodes, lr, gamma, tau,
        epsilon_start, epsilon_end, 
        decay_rate, batch_size, buffer_size
    ):

    wandb_run = wandb.init(
        dir='./output/',
        project='drl_hw1',
        mode="online",
        name=datetime.now().strftime("%Y.%m.%d-%H.%M.%S"),
    )
    wandb.config.update(
        {
            "output_dir": './output/',
        }
    )

    # agent_x, agent_y, direction, key possession, door status, actions
    policy = DQNAgent(gamma=gamma, tau=tau, lr=lr, checkpoint=None)
    policy.to('cuda')
    manager = TrainingManager(episodes=num_episodes)
    buffer = ReplayBuffer(buffer_size=buffer_size)

    epsilon = epsilon_start

    for episode in range(num_episodes):
        env = DynamicTaxiEnv(grid_size=random.randint(5, 10), fuel_limit=500)
        obs, _ = env.reset()
        policy.reset()

        observation = policy.get_observation(obs)  # Initially, the door is closed.

        done = False
        total_reward = 0
        episode_step = 0
        update_log=None

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
                update_log = policy.update(batch)

            # Move to the next state.
            observation = next_observation

        manager.add_episode(total_reward=total_reward, episode_step=episode_step)
        epsilon = max(epsilon_end, epsilon * decay_rate)

        wandb_run.log({
            'epsilon': epsilon,
            'total_reward': total_reward,
            'loss': update_log['loss'] if update_log is not None else 0.0,
            'episode_steps': episode_step,
            'episode': episode
        })

        manager.print_status(epsilon=epsilon)

    policy.save_checkpoint()
    
    plt.plot(manager.rewards_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Taxi DQN Training Progress")
    plt.savefig("training_reward.png")

if __name__ == '__main__':
    train_dqn()