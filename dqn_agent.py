import numpy as np
import pickle
import torch
import torch.nn as nn
import random

class QNetwork(nn.Sequential):

    def __init__(self, obs_dim, action_dim):
        layers = [
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ]
        super(QNetwork, self).__init__(*layers)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        return super().forward(x)

class DQNAgent:

    def __init__(self, gamma, tau, lr, checkpoint=None):
        self.observation_space = (
            1,      # stage: go to passenger, go to goal
            19,     # diff_x: (10-1)*2 + 1
            19,     # diff_y: (10-1)*2 + 1
            1,      # obstacle north
            1,      # obstacle south
            1,      # obstacle east
            1,      # obstacle west
        )
        self.action_space = (6,)
        self.obs_dim = sum(self.observation_space)
        self.action_dim = sum(self.action_space)
        self.q_network = QNetwork(obs_dim=self.obs_dim, action_dim=self.action_dim)
        self.q_target_network = QNetwork(obs_dim=self.obs_dim, action_dim=self.action_dim)
        self.q_network.to('cuda')
        self.q_target_network.to('cuda')

        self.gamma = gamma
        self.tau = tau

        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.q_network.parameters(), lr=lr, amsgrad=True)

        if checkpoint:
            self.load_checkpoint(checkpoint)

        self.reset()

    def reset(self):
        self.stage = 0
        self.current_target = 0

    def get_observation(self, obs):
        """
        obs = (
            taxi_row, 
            taxi_col, 
            self.stations[0][0],
            self.stations[0][1],
            self.stations[1][0],
            self.stations[1][1],
            self.stations[2][0],
            self.stations[2][1],
            self.stations[3][0],
            self.stations[3][1],
            obstacle_north, 
            obstacle_south, 
            obstacle_east, 
            obstacle_west, 
            passenger_look, 
            destination_look
        )
        """
        # Set stage 

        # observation: stage, one_hot_diff_x, one_hot_diff_y, obstacle 4 directions
        station_idx = self.current_target * 2 + 2
        station_row = obs[station_idx]
        station_col = obs[station_idx+1]

        # add 9 to shift to postive
        diff_x = station_row - obs[0] + 9 # taxi_row
        diff_y = station_col - obs[1] + 9 # taxi_col

        stage_tensor = torch.tensor([self.stage], dtype=torch.float32)
        diff_x_tensor = torch.nn.functional.one_hot(torch.tensor(diff_x), num_classes=self.observation_space[1]).type(torch.float32)
        diff_y_tensor = torch.nn.functional.one_hot(torch.tensor(diff_y), num_classes=self.observation_space[2]).type(torch.float32)

        obstacle_tensor = torch.tensor(obs[10:14], dtype=torch.float32)

        obs_tensor = torch.cat((stage_tensor, diff_x_tensor, diff_y_tensor, obstacle_tensor), dim=0)
        return obs_tensor

    def predict_action(self, obs, epislon):
        if np.random.uniform(0, 1) > epislon:
            with torch.no_grad():
                q_values = self.q_network(obs.unsqueeze(dim=0).to('cuda'))
                action = torch.argmax(q_values).item()
        else:
            action = random.choice(list(range(6)))

        return action

    def update(self, batch):
        obs_batch = batch['obs'].to('cuda')
        action_batch = batch['action'].to('cuda')
        reward_batch = batch['reward'].to('cuda')
        next_obs_batch = batch['next_obs'].to('cuda')
        # done_batch = batch['done'].to('cuda')

        next_state_values = torch.zeros(action_batch.shape, dtype=torch.float32, device='cuda')
        # non_final_mask = done_batch.to('cuda')
        with torch.no_grad():
            output = self.q_target_network(next_obs_batch)
            next_state_values = torch.max(output, dim=1)[0]

        state_action_values = self.q_network(obs_batch).gather(1, action_batch)
        expected_state_action_values = reward_batch + (next_state_values.unsqueeze(1) * self.gamma)
        
        loss = self.loss_fn(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()

        q_target_network_state_dict = self.q_target_network.state_dict()
        q_network_state_dict = self.q_network.state_dict()
        for key in q_network_state_dict:
            q_target_network_state_dict[key] = q_network_state_dict[key]*self.tau + q_target_network_state_dict[key]*(1-self.tau)

        self.q_target_network.load_state_dict(q_target_network_state_dict)

    def shaped_reward(self, obs, next_obs):
        return 0
    
    def save_checkpoint(self, checkpoint='taxi-dqn-agent.pt'):
        torch.save(self.q_network.state_dict(), open(checkpoint, 'ab'))

    def load_checkpoint(self, checkpoint='taxi-q-learning-agent.pkl'):
        q_network_state_dict = torch.load(open(checkpoint, 'rb'))
        self.q_network.load_state_dict(q_network_state_dict)