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
            1,     # diff_x: [-9, 9] normalized
            1,     # diff_y: [-9, 9] normalized
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
        self.last_action = -1
        self.passenger_location = None
        self.destination_location = None

    def get_observation(self, obs):
        """
        obs = (
            taxi_row,               0
            taxi_col,               1
            self.stations[0][0],    2
            self.stations[0][1],    3
            self.stations[1][0],    4
            self.stations[1][1],    5
            self.stations[2][0],    6
            self.stations[2][1],    7
            self.stations[3][0],    8
            self.stations[3][1],    9
            obstacle_north,         10
            obstacle_south,         11
            obstacle_east,          12
            obstacle_west,          13
            passenger_look,         14
            destination_look        15
        )
        """
        # Set stage and current_target
        passenger_look = obs[14]
        destination_look = obs[15]
        taxi_pos = obs[:2]
        current_target_pos = obs[self.current_target * 2 + 2: self.current_target * 2 + 4]
        if self.stage == 0 and self.last_action == 4 and passenger_look and current_target_pos == taxi_pos:
            self.stage = 1
        if self.stage == 1 and self.last_action == 5 and not(destination_look and current_target_pos == taxi_pos):
            self.stage = 0
            self.passenger_location = taxi_pos

        dist = abs(current_target_pos[0] - taxi_pos[0]) + abs(current_target_pos[1] - taxi_pos[1])
        if dist <= 1:
            if passenger_look:
                self.passenger_location = current_target_pos
            elif destination_look:
                self.destination_location = current_target_pos
            else:
                self.current_target = (self.current_target + 1) % 4

        # observation: stage, one_hot_diff_x, one_hot_diff_y, obstacle 4 directions
        if self.stage == 0 and self.passenger_location:
            current_target_pos = self.passenger_location
        elif self.stage == 1 and self.destination_location:
            current_target_pos = self.destination_location
        else:
            current_target_pos = obs[self.current_target * 2 + 2: self.current_target * 2 + 4]

        # add 9 to shift to postive
        stage_tensor = torch.tensor([self.stage], dtype=torch.float32)

        diff_x = (current_target_pos[0] - taxi_pos[0]) / 9.0 # taxi_row, normalize
        diff_y = (current_target_pos[1] - taxi_pos[1]) / 9.0 # taxi_col, normalize

        # diff_x_tensor = torch.nn.functional.one_hot(torch.tensor(diff_x), num_classes=self.observation_space[1]).type(torch.float32)
        # diff_y_tensor = torch.nn.functional.one_hot(torch.tensor(diff_y), num_classes=self.observation_space[2]).type(torch.float32)
        diff_tensor = torch.tensor([diff_x, diff_y], dtype=torch.float32)

        obstacle_tensor = torch.tensor(obs[10:14], dtype=torch.float32)

        obs_tensor = torch.cat((stage_tensor, diff_tensor, obstacle_tensor), dim=0)
        return obs_tensor

    def predict_action(self, obs, epislon, device='cuda'):
        if np.random.uniform(0, 1) > epislon:
            with torch.no_grad():
                q_values = self.q_network(obs.unsqueeze(dim=0).to(device))
                action = torch.argmax(q_values).item()
        else:
            action = random.choice(list(range(6)))

        self.last_action = action
        return action

    def update(self, batch):
        obs_batch = batch['obs'].to('cuda')
        action_batch = batch['action'].to('cuda')
        reward_batch = batch['reward'].to('cuda')
        next_obs_batch = batch['next_obs'].to('cuda')
        done_batch = batch['done'].to('cuda')

        next_state_values = torch.zeros(action_batch.shape, dtype=torch.float32, device='cuda')
        with torch.no_grad():
            next_actions = self.q_network(next_obs_batch).argmax(dim=1, keepdim=True)
            next_state_values = self.q_target_network(next_obs_batch).gather(1, next_actions)

        state_action_values = self.q_network(obs_batch).gather(1, action_batch)
        expected_state_action_values = reward_batch + (next_state_values * self.gamma) * (1 - done_batch.float())
        
        loss = self.loss_fn(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        q_target_network_state_dict = self.q_target_network.state_dict()
        q_network_state_dict = self.q_network.state_dict()
        for key in q_network_state_dict:
            q_target_network_state_dict[key] = q_network_state_dict[key]*self.tau + q_target_network_state_dict[key]*(1-self.tau)

        self.q_target_network.load_state_dict(q_target_network_state_dict)
        return {'loss': loss.item()}

    def shaped_reward(self, obs, next_obs):
        diff_x = obs[1].item() * 9 # unnormalize
        diff_y = obs[2].item() * 9
        next_diff_x = next_obs[1].item() * 9
        next_diff_y = next_obs[2].item() * 9
        
        current_dist = abs(diff_x) + abs(diff_y)
        next_dist = abs(next_diff_x) + abs(next_diff_y)
        
        # Encourage reducing distance to target (passenger or destination)
        if next_dist < current_dist:
            return 0.6  # Small positive reward for progress
        elif next_dist > current_dist:
            return -0.3  # Small penalty for moving away
        return 0
    
    def save_checkpoint(self, checkpoint='taxi-dqn-agent.pt'):
        torch.save(self.q_network.state_dict(), open(checkpoint, 'wb'))

    def load_checkpoint(self, checkpoint='taxi-q-learning-agent.pkl'):
        q_network_state_dict = torch.load(open(checkpoint, 'rb'), map_location=torch.device('cpu'))
        self.q_network.load_state_dict(q_network_state_dict)

    def to(self, device):
        self.q_network.to(device)
        self.q_target_network.to(device)