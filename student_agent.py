# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

from dqn_agent import DQNAgent

policy = DQNAgent(0, 0, 0, 'taxi-dqn-agent.pt')
policy.to('cpu')

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.   
    # print(obs)
    observation = policy.get_observation(obs)
    action = policy.predict_action(observation, 0.0, 'cpu')
    return action