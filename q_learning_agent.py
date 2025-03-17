import numpy as np
import pickle

class QLearningAgent:

    def __init__(self, env, alpha, gamma, ckpt_name=None):
        self.q_table = np.zeros(
            (
                5, # taxi row
                5, # taxi col
                5, # passenger location
                4, # destination
                6, # action
            )
        )
        self.alpha = alpha
        self.gamma = gamma

        if ckpt_name:
            self.load_checkpoint(ckpt_name)

    def get_state(self, obs):
        destination = obs % 4
        obs //= 4
        passenger_location = obs % 5
        obs //= 5
        taxi_col = obs % 5
        obs //= 5
        taxi_row = obs
        return np.array([taxi_row, taxi_col, passenger_location, destination])

    def predict_action(self, state, epislon):
        if np.random.uniform(0, 1) > epislon:
            action = np.argmax(self.q_table[state[0], state[1], state[2], state[3]])
        else:
            action = np.random.choice(6, 1)[0]

        return action

    def update_q_table(self, state, action, reward, next_state):
        self.q_table[
            state[0], 
            state[1], 
            state[2], 
            state[3], 
            action
        ] += self.alpha * (reward + self.gamma * np.max(
                self.q_table[
                    next_state[0],
                    next_state[1],
                    next_state[2],
                    next_state[3],
                ]
            ) - self.q_table[
                    state[0], 
                    state[1], 
                    state[2], 
                    state[3], 
                    action
                ]
        )

    def reset(self, env):
        pass

    def shaped_reward(self, state, next_state, gamma, termination):
        return 0
    
    def save_checkpoint(self, checkpoint='taxi-q-learning-agent.pkl'):
        pickle.dump(self.q_table, open(checkpoint, 'ab'))

    def load_checkpoint(self, checkpoint='taxi-q-learning-agent.pkl'):
        self.q_table = pickle.load(open(checkpoint, 'rb'))