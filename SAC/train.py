import sys
sys.path.append("/users/aizam/cart-pole/") #changeable

from cartpole.simulator import Simulator, State, Error, Target
from cartpole import log

from trainer import Trainer, Config

from utils import compute_reward

from itertools import count

import tqdm
import torch
import numpy as np


N_EPOCHS = 2
N_EPOCHS_NOISED = 5


config = Config(
    state_dim=6,
    action_dim=1,
    max_buffer_size = 25000,
    device_name = 'cpu',
    batch_size = 768,
    actor_lr = 1e-4,
    critic_lr = 1e-4,
    tau = 0.005,
    max_grad_norm = 10,
    alpha = 0.1, #the smaller the action space, the smaller should alpha be (entropy regularization)
    gamma = 0.99,
    start_timesteps = 0,
    timesteps_per_epoch = 1,
    policy_update_freq = 1,
    opt_eps = 1e-4,
    min_std = 0.1,
    max_std = 8,
    delta = 0.02,
    n_iterations=2000,
    n_epochs=10,
    checkpoint_freq=1000
)


log.setup(log_path='training_log.mcap', level=log.Level.DEBUG)

env = Simulator()
trainer = Trainer(log, config)

interaction_state = State(
        cart_position=0,
        pole_angle=0,
        cart_velocity=0,
        angular_velocity=0,
        )

reset_state = State(
        cart_position=0,
        pole_angle=0,
        cart_velocity=0,
        angular_velocity=0,
        )

for episode in range(1, 100):

    print(interaction_state.stamp)
    print(episode)
    
    if episode % 2 == 0:
        
        noised_position = np.random.uniform(low=0, high=0.15)

        noised_pole_angle = np.random.uniform(low=2, high=3.14)
        
        reset_state = State(
            cart_position=noised_position,
            pole_angle=noised_pole_angle,
            cart_velocity=0,
            angular_velocity=0,
            )
        
        interaction_state = State(
            cart_position=noised_position,
            pole_angle=noised_pole_angle,
            cart_velocity=0,
            angular_velocity=0,
            )
    
        
    if episode % 2 != 0:
        
        reset_state = State(
            cart_position=0,
            pole_angle=3.14,
            cart_velocity=0,
            angular_velocity=0,
            )
        
        interaction_state = State(
            cart_position=0,
            pole_angle=3.14,
            cart_velocity=0,
            angular_velocity=0,
            )
    
    env.reset(state=interaction_state)
        
    trainer.train(env, interaction_state, reset_state, episode)

        
