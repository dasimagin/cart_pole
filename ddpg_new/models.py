import torch
from torch import nn

from utils import ExperimentConfig

class Actor(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super(Actor, self).__init__()
        self.max_action = config.max_action
        layers = []
        for _ in range(config.critic_layers):
            layers.append(nn.Linear(config.actor_width, config.actor_width))
            layers.append(nn.LeakyReLU())

        self.q1 = nn.Sequential(
            nn.Linear(config.state_dim, config.actor_width),
            nn.LeakyReLU(),
            *layers,
            nn.Linear(config.actor_width, config.action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.q1(state) * self.max_action



class Critic(nn.Module):
    def __init__(self, config: ExperimentConfig):
        super(Critic, self).__init__()
        layers = [ ]
        for _ in range(config.critic_layers):
            layers.append(nn.Linear(config.critic_width, config.critic_width))
            layers.append(nn.LeakyReLU())

        self.q1 = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.critic_width),
            nn.LeakyReLU(),
            *layers,
            nn.Linear(config.critic_width, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.critic_width),
            nn.LeakyReLU(),
            *layers,
            nn.Linear(config.critic_width, 1)
        )

    def forward(self, state, action):
        inp = torch.cat([state, action], 1)
        return self.q1(inp), self.q2(inp)
