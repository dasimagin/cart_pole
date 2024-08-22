import torch
import torch.nn as nn

import numpy as np
from math import cos

from cartpole.common import State, BaseModel, Error
from cartpole.simulator import Target
from cartpole import log

import time

class Scalar(BaseModel):
    value: float

def play_and_record(initial_state, agent, env, exp_replay, delta, start_state, n_steps=10):
    state = initial_state
    sum_rewards = 0
    time.sleep(0.02)

    for t in range(n_steps):

        prev_state = state
        env.advance(delta)
        state = env.get_state()
        action = agent.get_action(state)
        reward = compute_reward(state, action)

        exp_replay.add(prev_state, state, action, reward.value)

        last_element = exp_replay.get_last()
        if last_element is not None:
            terminated = last_element[-1]
            
        env.set_target(Target(acceleration=action))
        
        if terminated or state.stamp > 7 or np.abs(state.pole_angle) > np.pi + 0.1:
            reward = Scalar(value=-1)
            termination_state = state
            log.publish('/cartpole/state', state, state.stamp)
            log.publish('/cartpole/reward', reward, state.stamp)
            state = start_state
            env.reset(state=state)
            time.sleep(0.02)
            return reward.value, termination_state

        sum_rewards += reward.value
        
        log.publish('/cartpole/state', state, state.stamp)
        log.publish('/cartpole/reward', reward, state.stamp)

    return sum_rewards, state


def optimize(name, model, optimizer, loss, state: State, env, max_grad_norm=10):

    loss = loss.mean()
    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    
    state_stmp = env.get_state()
    loss_log = Scalar(value=loss.item())
    loss_name = 'loss' + name
    grad_name = 'grad_norm' + name
    log.publish(loss_name, loss_log, state_stmp.stamp)
    grad_norm_log = Scalar(value=grad_norm.item())
    log.publish(grad_name, grad_norm_log, state_stmp.stamp)
    
    

def update_target_networks(model, target_model, tau):
    for param, target_param in zip(model.parameters(), target_model.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def compute_reward(state: State):
    angle_reward = 1-(cos(state.pole_angle) + 0.01 * state.pole_angular_velocity**2 )
    straight_up_reward = 0.1 if cos(state.pole_angle) < 0.1 else 0
    position_reward = -state.cart_position**2
    return angle_reward + position_reward + straight_up_reward

def play_and_record_evaluate(initial_state, agent, env, delta, start_state, n_steps=10):
    # added the start_state to include diversity because we will change it 
    state = initial_state
    sum_rewards = 0

    for t in range(n_steps):
        prev_state = state
        env.advance(delta)
        state = env.get_state()
        action = agent.get_action(state)
        reward = compute_reward(state, action)
        env.set_target(Target(acceleration=action))
        
        if state.error == Error.NO_ERROR:
            state = start_state
            env.reset(state=state)
        else:
            state = state

        sum_rewards += reward.value
        
        log.publish('/cartpole/state', state, state.stamp)
        log.publish('/cartpole/reward', reward, state.stamp)

    return sum_rewards, state
        