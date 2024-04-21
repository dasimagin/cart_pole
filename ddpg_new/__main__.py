import sys
sys.path.append("/Users/severovv/hse/cart-pole/cart-pole")

from cartpole.simulator import Simulator, State, Error, Target
from cartpole import log

from trainer import Trainer, Config, compute_reward, Scalar

from itertools import count

import pathlib
import datetime
import tqdm
import torch

config = Config(
    state_dim = 5,
    action_dim = 1,
    max_action = 5,
    discount = 0.99,
    memory_size = int(1e6),
    device_name = 'mps',
    batch_size = 1024,
    actor_lr = 1e-5,
    actor_grad_norm=0.1,
    actor_width=1024,
    critic_lr = 3e-4,
    critic_grad_norm=0.1,
    critic_noise=0.1,
    critic_width=1024,
    tau = 0.01,
    finish_penalty=0
)

delta = 0.02
total = 0

logdir = pathlib.Path.cwd() / "trainings" / datetime.datetime.now().ctime()
if not logdir.exists():
    logdir.mkdir(parents=True)


log.setup(log_path=str(logdir / 'training.mcap'), level=log.Level.DEBUG)
cartpole = Simulator(integration_step=delta/20)
trainer = Trainer(log, config)
state = State()

rewards = []

for episode in tqdm.tqdm(count()):
    state = State(
        cart_position=torch.randn((1,)).item()*0.2,
        pole_angle=torch.randn((1, )).item()*0.5
    )

    cartpole.reset(state=state)
    total_reward = 0

    for t in tqdm.tqdm(range(500), leave=False):
        total += 1
        prev_state = state

        if episode > 5:
            action = trainer.select_action(state, sigma=0.1)
        else:
            action = torch.tensor(0)

        trainer.log_predictions(state, total)
        action = action.item()

        cartpole.set_target(Target(acceleration=action))

        cartpole.advance(delta)
        state = cartpole.get_state()
        reward = compute_reward(state)
        if state.error:
            reward = -10

        total_reward = value=total_reward + reward
        done = (state.error != Error.NO_ERROR) or (t == 499)

        trainer.memory.add(prev_state, state, action, reward, done)

        batch = trainer.memory.sample(trainer.config.batch_size, trainer.device)
        if len(batch) > config.batch_size:
            trainer.train_critic(batch, total)
            trainer.train_actor(batch, total)
            trainer.soft_update()

        log.publish('/cartpole/state', state, total)
        log.publish('/cartpole/reward', Scalar(value=reward), total)
        log.publish('/cartpole/episode_reward', Scalar(value=total_reward), total)
        log.publish('cartpole/ten_reward_mean', Scalar(value=torch.mean(torch.tensor(rewards[-10:])).item()))

        if done:
            rewards.append(total_reward)
            break
