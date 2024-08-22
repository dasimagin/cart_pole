import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from torch.distributions import Normal

from replay_buffer import state_to_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# LOG_STD_MIN, LOG_STD_MAX = -20, 2

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.h = 2056  #change
        self.action_dim = action_dim

        self.actor_model = nn.Sequential(
            nn.Linear(in_features=state_dim, out_features=self.h),
            nn.ReLU(),
            nn.Linear(in_features=self.h, out_features=self.h),
            nn.ReLU(),
            nn.Linear(in_features=self.h, out_features=self.h),
            nn.ReLU(),
            nn.Linear(in_features=self.h, out_features=self.h),
            nn.ReLU(),
            nn.Linear(in_features=self.h, out_features=action_dim*2)
        )

    def apply(self, states):
        m = -20
        M = 2
        states = states.to(DEVICE)
        output = self.actor_model(states)
        means = output[..., :self.action_dim]
        var = torch.tanh(output[..., self.action_dim:]) + 1
        var = 0.5 * (M - m) * var + m
        var = torch.exp(var)
        normal_distr = Normal(means, var)

        actions_first = normal_distr.rsample()
        actions = torch.tanh(actions_first)
        log_prob = normal_distr.log_prob(actions_first) - torch.log(1 - actions**2 + 1e-6)

        #this is a more numerically stable version of the appendix C eq.21 https://arxiv.org/pdf/1801.01290.pdf 

        return actions, log_prob

    def get_action(self, states):

        with torch.no_grad():
            
            states = state_to_tensor(states)
            actions, _ = self.apply(states)
            actions = actions.cpu().detach().numpy()
            
            assert isinstance(actions, (list,np.ndarray))
            assert actions.max() <= 1. and actions.min() >= -1, "actions must be in the range [-1, 1]"
            return actions.item()


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.h = 2056
        input_dim = state_dim + action_dim
        self.critic_model = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=self.h),
            nn.ReLU(),
            nn.Linear(in_features=self.h, out_features=self.h),
            nn.ReLU(),
            nn.Linear(in_features=self.h, out_features=self.h),
            nn.ReLU(),
            nn.Linear(in_features=self.h, out_features=1)
        )

    def get_qvalues(self, states, actions):

        batch = torch.cat([states, actions], dim=1)
        qvalues = self.critic_model(batch)
        
        # assert len(qvalues.shape) == 1 and qvalues.shape[0] == states.shape[0]
        return qvalues