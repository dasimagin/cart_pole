import sys
sys.path.append("/Users/severovv/hse/cart-pole/cart-pole")

from cartpole.simulator import Simulator, State, Error, Target, Config, Limits, Parameters
from cartpole import log
from math import pi

from trainer import Trainer, ExperimentConfig, compute_reward, Scalar

import pathlib
import datetime
import tqdm
import torch

from matplotlib import pyplot as plt


def run_experiment(config: ExperimentConfig):
    logdir = pathlib.Path.cwd() / "trainings" / datetime.datetime.now().ctime()
    if not logdir.exists():
        logdir.mkdir(parents=True)
    log.setup(log_path=str(logdir / 'training.mcap'), level=log.Level.DEBUG)

    cartpole = Simulator()
    cartpole.set_config(Config(
            hardware_limit=Limits(
                cart_position=0.5, cart_velocity=2.0, cart_acceleration=5.0
            ),
            control_limit=Limits(),
            parameters=Parameters(gravity=9.81, friction_coef=0.1, mass_coef=4),
        )
    )

    trainer = Trainer(log, config)
    state = State()

    rewards = []
    total = 0

    for episode in tqdm.tqdm(range(config.episodes_count)):
        state = State(
            cart_position=torch.randn((1,)).item()*0.3,
            pole_angle= torch.randn((1,)).item()*pi/4,
            pole_angular_velocity=torch.randn((1,)).item()*0.01
        )

        cartpole.reset(state=state)
        total_reward = 0

        for t in tqdm.tqdm(range(1000), leave=False):
            total += 1
            prev_state = state

            if episode < config.warmup_stay:
                action = torch.tensor(0)
            elif episode < config.warmup_stay + config.warmup_random:
                action = config.max_action * (2*torch.rand((1,)) - 1) / 5
            else:
                action = trainer.select_action(state, sigma=config.action_noise)


            trainer.log_predictions(state, total)
            action = action.item()

            cartpole.set_target(Target(acceleration=action))
            cartpole.advance(config.delta)
            state = cartpole.get_state()

            done = (state.error != Error.NO_ERROR) or (abs(state.pole_angle) > pi*4)
            if done:
                reward = config.finish_penalty
            else:
                reward = compute_reward(state)
            total_reward += reward

            trainer.memory.add(prev_state, state, action, reward, done)

            if total % 3 == 0 and len(trainer.memory) >= config.batch_size:
                batch = trainer.memory.sample(trainer.config.batch_size, trainer.device)
                if episode > config.critic_start:
                    trainer.train_critic(batch, total)
                if episode > config.actor_start:
                    trainer.train_actor(batch, total)
                if episode > min(config.actor_start, config.critic_start):
                    trainer.soft_update()

            log.publish('/cartpole/state', state, datetime.datetime.now().timestamp())
            log.publish('/cartpole/reward', Scalar(value=reward), datetime.datetime.now().timestamp())
            log.publish('/cartpole/episode_reward', Scalar(value=total_reward), datetime.datetime.now().timestamp())
            log.publish('cartpole/ten_reward_mean', Scalar(value=torch.mean(torch.tensor(rewards[-10:])).item()), datetime.datetime.now().timestamp())

            if done:
                rewards.append(total_reward)
                break
    return rewards

run_experiment(
    ExperimentConfig(
        action_noise=0.01,
        warmup_random=10,
        warmup_stay=2,
        critic_start=2,
        actor_start=12,
        state_dim = 6,
        action_dim = 1,
        max_action = 3,
        discount = 0.99,
        memory_size = int(1e5),
        device_name = 'mps',
        batch_size = 1024,
        actor_lr = 1e-4,
        actor_grad_norm=1.0,
        actor_width=256,
        actor_layers=3,
        critic_grad_norm=1.0,
        critic_noise=0.1,
        critic_width=256,
        critic_layers=3,
        critic_lr = 1e-3,
        tau = 0.05,
        finish_penalty=-10,
        episodes_count=200,
        delta=0.02,
    )
)
