import collections
import numpy as np
import torch
import random

from math import pi, cos, tanh, sin

from cartpole import State, Error



def state_to_tensor(state: State):
    return torch.tensor([
        (state.cart_position / 0.25),
        (state.cart_velocity / 5.0),
        (state.cart_acceleration / 7.5),
        cos(state.pole_angle),
        state.pole_angle / (6 * np.pi), #important
        state.pole_angular_velocity / (6 * np.pi)  #change and normalize
    ])

def make_tensor(from_state, to_state, action, reward):
    return torch.concat([
        state_to_tensor(from_state),
        state_to_tensor(to_state),
        torch.tensor([action]),
        torch.tensor([reward]),
        torch.tensor([to_state.error != Error.NO_ERROR])
    ])


class ReplayBuffer:

    def __init__(self, state_dim: int, maxlen: int):
        self.state_dim = state_dim
        self.states = torch.zeros((maxlen, self.state_dim*2 + 3))
        self.maxlen = maxlen
        self.ptr = 0
        self.length = 0

    def __len__(self):
        return len(self.states)

    def add(self, from_state: State, to_state: State, action: float, reward: float):
        self.states[self.ptr] = make_tensor(from_state, to_state, action, reward)
        self.ptr = (self.ptr + 1) % self.maxlen
        self.length = min(self.length + 1, self.maxlen)
    
    def sample(self, sample_size: int, device):
        sample_size = min(self.length, sample_size)
        sample = self.states[random.sample(range(self.length), sample_size)]
        return (
            sample[:, :self.state_dim].to(device),
            sample[:, self.state_dim:self.state_dim*2].to(device),
            sample[:, self.state_dim*2].reshape(-1, 1).to(device),
            sample[:, self.state_dim*2+1].reshape(-1, 1).to(device),
            sample[:, self.state_dim*2+2].reshape(-1, 1).to(device)
        )
    def get_last(self):
        if len(self.states) == 0:
            return None
        return self.states[(self.ptr - 1) % self.maxlen]