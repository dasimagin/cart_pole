import torch
from torch import nn

from utils import Config

class Actor(nn.Module):
    def __init__(self, config: Config):
        super(Actor, self).__init__()
        self.max_action = config.max_action
        self.q1 = nn.Sequential(
              nn.Linear(config.state_dim, 256),
              nn.ReLU(),
              nn.BatchNorm1d(256),
              nn.Linear(256, 1024),
              nn.ReLU(),
              nn.BatchNorm1d(1024),
              nn.Linear(1024, 512),
              nn.ReLU(),
              nn.BatchNorm1d(512),
              nn.Linear(512, 256),
              nn.ReLU(),
              nn.BatchNorm1d(256),
              nn.Linear(256, config.action_dim),
              nn.Tanh()
        )

    def forward(self, state):
        return self.q1(state) * self.max_action
    


class Critic(nn.Module):
    def __init__(self, config: Config):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),  
            nn.Linear(512, 1024),
            nn.ReLU(),  
            nn.Linear(1024, 512),
            nn.ReLU(),  
            nn.Linear(512, 256),
            nn.ReLU(),  
            nn.Linear(256, 1),
        )

        self.q2 = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),  
            nn.Linear(512, 1024),
            nn.ReLU(),  
            nn.Linear(1024, 512),
            nn.ReLU(),  
            nn.Linear(512, 256),
            nn.ReLU(),  
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def get_q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)