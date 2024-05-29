from cartpole import State
from utils import ExperimentConfig, make_tensor

import random
import torch


class ReplayMemory:
    def __init__(self, config: ExperimentConfig):
        self.state_size = config.state_dim
        self.states = torch.zeros((config.memory_size, self.state_size*2 + 3))
        self.maxlen = config.memory_size
        self.ptr = 0
        self.length = 0
        self.config = config

    def add(self, from_state: State, to_state: State, action: float, reward: float, done: bool):
        self.states[self.ptr] = make_tensor(from_state, to_state, action, reward, self.config, done)
        self.ptr = (self.ptr + 1) % self.maxlen
        self.length = min(self.length + 1, self.maxlen)

    def sample(self, sample_size: int, device):
        sample_size = min(self.length, sample_size)
        sample = self.states[random.sample(range(self.length), sample_size)]
        return (
            sample[:, :self.state_size].to(device),
            sample[:, self.state_size:self.state_size*2].to(device),
            sample[:, self.state_size*2].reshape(-1, 1).to(device),
            sample[:, self.state_size*2+1].reshape(-1, 1).to(device),
            sample[:, self.state_size*2+2].reshape(-1, 1).to(device)
        )

    def __len__(self):
        return self.length
