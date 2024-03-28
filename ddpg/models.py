import torch
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, max_action, width=128):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.nn = nn.Sequential(
              nn.Linear(state_size, width),
              nn.ReLU(),
              nn.Linear(width, width),
              nn.ReLU(),
              nn.Linear(width, width),
              nn.ReLU(),
              nn.Linear(width, action_size),
              nn.Tanh()
        )

    def forward(self, state):
        return self.nn(state) * self.max_action


class Critic(nn.Module):
    def __init__(self, state_size, action_size, width=128):
        super(Critic, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_size + action_size, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),  
            nn.Linear(width, 1),
        )


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)
