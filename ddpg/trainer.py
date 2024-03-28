from cartpole.common import State, BaseModel
from replay_memory import ReplayMemory, state_to_tensor
from models import Actor, Critic

from copy import deepcopy

import torch
import numpy as np

class Scalar(BaseModel):
    value: float

class Config(BaseModel):
    state_dim: int
    action_dim: int
    max_action: float
    discount: float
    memory_size: int
    device_name: str 
    batch_size: int
    actor_lr: float
    critic_lr: float
    tau: float


def compute_reward(state: State):
    value = 0.1 + 5*(1-np.cos(state.pole_angle))
    if abs(state.cart_position) > 0.35:
        value = -1
    return Scalar(value=value)

class Trainer:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.device = torch.device(config.device_name)

        self.memory = ReplayMemory(config.state_dim, config.memory_size)
        
        self.actor = Actor(config.state_dim, config.action_dim, config.max_action).to(self.device)
        self.critic = Critic(config.state_dim, config.action_dim).to(self.device)

        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)

        self.target_actor.eval()
        self.target_critic.eval()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr, weight_decay=1e-2)

        self.loss = torch.nn.MSELoss()

    def soft_update(self):
        self.do_soft_update(self.actor, self.target_actor)
        self.do_soft_update(self.critic, self.target_critic)

    def do_soft_update(self, source_net, target_net):
        for source_param, target_param in zip(source_net.parameters(), target_net.parameters()):
            target_param.data = self.config.tau * source_param.data + (1 - self.config.tau) * target_param.data
    
    
    def select_action(self, state: State, sigma: float, train: bool, episode: int, total: int):
        stat = state_to_tensor(state).reshape(1, -1).to(self.device)
        act = self.target_actor(stat)
        act += torch.normal(0.0, sigma, size=act.shape).to(self.device)
        pred_act = act

        if episode < 20:
            act = torch.normal(0.0, 0.5, size=(1, )).reshape(-1, 1).to(self.device)

        res = self.target_critic(stat, act)

        self.logger.publish("action", Scalar(value=pred_act.item()), total)
        self.logger.publish("pred", Scalar(value=res.item()), total)
        return act.clip(min=-self.config.max_action, max=self.config.max_action)


    def learn(self, total):
        if self.memory.length < self.config.batch_size:
            return

        states, next_states, actions, rewards, dones = self.memory.sample(self.config.batch_size, self.device)

        next_actions = self.target_actor(next_states)
        action_noise = torch.normal(0, 0.05, size=next_actions.shape).to(self.device)
        target_Q = self.target_critic(next_states, next_actions + action_noise) * (1-dones)
        target_Q = rewards + self.config.discount * target_Q

        current_Q = self.critic(states, actions)
        critic_loss = self.loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.1)
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()
        
        if total % 10 == 0:
            self.soft_update()