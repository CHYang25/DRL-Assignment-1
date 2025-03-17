import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

from q_learning_agent import QLearningAgent

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

def train_q_learning(
        env_name="Taxi-v3", 
        episodes=100000, alpha=0.1, gamma=0.99,
        epsilon_start=1.0, epsilon_end=0.01, 
        decay_rate=0.9999, reward_shaping=True,
        debug=False
    ):
    
    env = gym.make(env_name)

    # agent_x, agent_y, direction, key possession, door status, actions
    policy = QLearningAgent(env, alpha, gamma)
    manager = TrainingManager(episodes=episodes)

    epsilon = epsilon_start

    for episode in range(episodes):
        obs, _ = env.reset()
        policy.reset(env)
        state = policy.get_state(obs)  # Initially, the door is closed.

        done = False
        total_reward = 0
        episode_step = 0

        while not done:
            # TODO: Implement ε-greedy policy for action selection.
            action = policy.predict_action(state, epsilon)

            # Execute the selected action.
            obs, reward, termination, truncated, _ = env.step(action)
            done = termination or truncated
            next_state = policy.get_state(obs)
            episode_step += 1

            shaped_reward = policy.shaped_reward(state, next_state, gamma, termination)
            reward += shaped_reward
            total_reward += reward

            # TODO: Apply Q-learning update rule (Bellman equation).
            policy.update_q_table(state, action, reward, next_state)

            # Move to the next state.
            state = next_state

        manager.add_episode(total_reward=total_reward, episode_step=episode_step)
        epsilon = max(epsilon_end, epsilon * decay_rate)

        if (episode + 1) % 100 == 0:
            manager.print_status(epsilon=epsilon)

    env.close()
    policy.save_checkpoint()
    
    plt.plot(manager.rewards_per_episode)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Taxi Q-Learning Training Progress")
    plt.savefig("training_reward.png")

if __name__ == '__main__':
    train_q_learning()