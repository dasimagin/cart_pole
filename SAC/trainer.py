from cartpole.common import State, BaseModel
from SAC.replay_buffer import ReplayBuffer, state_to_tensor
from models import Actor, Critic
from utils import play_and_record, optimize, update_target_networks


import torch
import torch.nn as nn

import numpy as np

class Scalar(BaseModel):
    value: float

class Config(BaseModel):
    state_dim: int
    action_dim: int
    max_buffer_size: int
    device_name: str 
    batch_size: int
    actor_lr: float
    critic_lr: float
    tau: float
    max_grad_norm: int
    alpha: float
    gamma: float
    start_timesteps: int
    timesteps_per_epoch: int
    policy_update_freq: int
    opt_eps: float
    min_std: float
    max_std: float
    delta: float
    n_iterations: int
    checkpoint_freq: int



class Trainer:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.device = torch.device(config.device_name)

        self.exp_replay = ReplayBuffer(config.state_dim, config.max_buffer_size)
        
        self.actor = Actor(config.state_dim, config.action_dim).to(self.device)

        self.critic1 = Critic(config.state_dim, config.action_dim).to(self.device)
        self.critic2 = Critic(config.state_dim, config.action_dim).to(self.device)

        self.target_critic1 = Critic(config.state_dim, config.action_dim).to(self.device)
        self.target_critic2 = Critic(config.state_dim, config.action_dim).to(self.device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

    
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr, eps=config.opt_eps)
        self.opt_critic1 = torch.optim.Adam(self.critic1.parameters(), lr=config.critic_lr, eps=config.opt_eps)
        self.opt_critic2 = torch.optim.Adam(self.critic2.parameters(), lr=config.critic_lr, eps=config.opt_eps)
        
        
    def compute_critic_target(self, rewards, next_states, is_done):
        is_not_done = 1 - is_done
        with torch.no_grad():
            next_actions, log_prob = self.actor.apply(next_states)
            q_values_1 = self.target_critic1.get_qvalues(next_states, next_actions)
            q_values_2 = self.target_critic2.get_qvalues(next_states, next_actions)
            v_values = torch.min(q_values_1, q_values_2) - self.config.alpha * log_prob.sum(dim=1)
            critic_target = (rewards + self.config.gamma * v_values * is_not_done).squeeze(-1)

        return critic_target

    def compute_actor_loss(self, states):

        actions, log_prob = self.actor.apply(states)
        q_values_1 = self.critic1.get_qvalues(states, actions)
        q_values_2 = self.critic2.get_qvalues(states, actions)
        q_values = torch.min(q_values_1, q_values_2)

        assert actions.requires_grad, "actions must be differentiable with respect to policy parameters"

        actor_loss = -q_values.mean() + self.config.alpha * log_prob.mean()
        return actor_loss
    
    
    def train(self, env, interaction_state, start_state, epoch):

        for n_iterations in range(0, self.config.n_iterations, self.config.timesteps_per_epoch):
            _, interaction_state = play_and_record(interaction_state, self.actor, env, self.exp_replay, self.config.delta, start_state, self.config.timesteps_per_epoch)
            states, next_states, actions, rewards, is_done = self.exp_replay.sample(self.config.batch_size, self.config.device_name)
            critic_target = self.compute_critic_target(rewards, next_states, is_done)
            # losses
            critic1_qvalues = self.critic1.get_qvalues(states, actions)
            critic1_loss = (critic1_qvalues - critic_target) ** 2
            optimize("critic1", self.critic1, self.opt_critic1, critic1_loss, states, env, 10)

            critic2_qvalues = self.critic2.get_qvalues(states, actions)
            critic2_loss = (critic2_qvalues - critic_target) ** 2
            optimize("critic2", self.critic2, self.opt_critic2, critic2_loss, states, env, 10)

            if n_iterations % self.config.policy_update_freq == 0:
                actor_loss = self.compute_actor_loss(states)
                optimize("actor", self.actor, self.opt_actor, actor_loss, states, env, 10)

                # update target networks
                update_target_networks(self.critic1, self.target_critic1, self.config.tau)
                update_target_networks(self.critic2, self.target_critic2, self.config.tau)


    def train_with_new_actor(self, env, interaction_state, start_state, epoch):
        
        for n_iterations in range(0, self.config.n_iterations, self.config.timesteps_per_epoch):
            _, interaction_state = play_and_record(interaction_state, self.actor, env, self.exp_replay, self.config.delta, start_state, self.config.timesteps_per_epoch)
            states, next_states, actions, rewards, is_done = self.exp_replay.sample(self.config.batch_size, self.config.device_name)
            critic_target = self.compute_critic_target(rewards, next_states, is_done)
            # losses
            critic1_qvalues = self.critic1.get_qvalues(states, actions)
            critic1_loss = (critic1_qvalues - critic_target) ** 2
            optimize("critic1", self.critic1, self.opt_critic1, critic1_loss, states, self.config.max_grad_norm)

            critic2_qvalues = self.critic2.get_qvalues(states, actions)
            critic2_loss = (critic2_qvalues - critic_target) ** 2
            optimize("critic2", self.critic2, self.opt_critic2, critic2_loss, states, self.config.max_grad_norm)

            if n_iterations % self.config.policy_update_freq == 0:
                actor_loss = self.compute_actor_loss(states)
                optimize("actor", self.actor, self.opt_actor, actor_loss, states, self.config.max_grad_norm)

                # update target networks
                update_target_networks(self.critic1, self.target_critic1, self.config.tau)
                update_target_networks(self.critic2, self.target_critic2, self.config.tau)