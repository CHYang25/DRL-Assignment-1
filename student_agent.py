# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

from q_learning_agent import QLearningAgent

policy = QLearningAgent(None, 0, 0, 'taxi-q-learning-agent.pkl')

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    state = policy.get_state(obs)
    action = policy.predict_action(state, 0)
    return action