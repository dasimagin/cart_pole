
from replay_memory import ReplayMemory
from models import Actor, Critic
from cartpole import State
from utils import Scalar, Config, state_to_tensor
from cartpole.log import Logger

from copy import deepcopy

import torch
import numpy as np



def compute_reward(state: State):
    value = (1 - np.cos(state.pole_angle))
    value += (0.4 - abs(state.cart_position))/10
    return value

class Trainer:
    def __init__(self, logger: Logger, config: Config):
        self.logger = logger
        self.config = config
        self.device = torch.device(config.device_name)

        self.memory = ReplayMemory(config)
        
        self.actor, self.critic = Actor(config).to(self.device), Critic(config).to(self.device)
        self.target_actor, self.target_critic = deepcopy(self.actor), deepcopy(self.critic)

        self.actor_optimizer = torch.optim.RAdam(self.actor.parameters(), lr=config.actor_lr, weight_decay=1e-2)
        self.critic_optimizer = torch.optim.RAdam(self.critic.parameters(), lr=config.critic_lr)

        self.loss = torch.nn.MSELoss()

    def soft_update(self):
        self.do_soft_update(self.actor, self.target_actor)
        self.do_soft_update(self.critic, self.target_critic)

    def do_soft_update(self, source_net, target_net):
        for source_param, target_param in zip(source_net.parameters(), target_net.parameters()):
            target_param.data = self.config.tau * source_param.data + (1 - self.config.tau) * target_param.data
    
    
    def select_action(self, state: State, sigma: float):
        self.actor.eval()
        stat = state_to_tensor(state, self.config).reshape(1, -1).to(self.device)
        with torch.no_grad():
            act = self.actor(stat)
        act += torch.randn(act.shape).to(self.device)*sigma
        self.actor.train()
        return act.clip(min=-self.config.max_action, max=self.config.max_action)
    
    def log_predictions(self, state: State, total: int):
        self.actor.eval()
        self.critic.eval()
        stat = state_to_tensor(state, self.config).reshape(1, -1).to(self.device)
        with torch.no_grad():
            act = self.actor(stat)
            eval1, eval2 = self.critic(stat, act)
        self.logger.publish("action", Scalar(value=act.item()), total)
        self.logger.publish("pred_1", Scalar(value=eval1.item()), total)
        self.logger.publish("pred_2", Scalar(value=eval2.item()), total)
        self.actor.train()
        self.critic.train()

    def train_critic(self, batch, total: int):
        states, next_states, actions, rewards, dones = batch

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            action_noise = torch.randn(next_actions.shape).to(self.device)*self.config.critic_noise
            target_Q1, target_Q2 = self.target_critic(next_states, next_actions + action_noise)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + self.config.discount * target_Q * (1 - dones)

        current_Q1, current_Q2 = self.critic(states, actions)
        critic_loss = self.loss(current_Q1, target_Q) + self.loss(current_Q2, target_Q)
        self.logger.publish("critic_loss", Scalar(value=critic_loss.mean().item()), total)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.critic_grad_norm)
        self.critic_optimizer.step()
    
    def train_actor(self, batch, total: int):
        states, next_states, actions, rewards, dones = batch
        actor_loss = -self.critic.get_q1(states, self.actor(states)).mean()
        self.logger.publish("actor_loss", Scalar(value=actor_loss.mean().item()), total)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.actor_grad_norm)
        self.actor_optimizer.step()