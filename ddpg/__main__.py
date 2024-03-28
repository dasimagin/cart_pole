import sys
sys.path.append("/Users/severovv/hse/cart-pole/cart-pole")

from cartpole.simulator import Simulator, State, Error, Target
from cartpole import log

from trainer import Trainer, Config, compute_reward, Scalar

from itertools import count

import tqdm
import torch

config = Config(
    state_dim = 5,
    action_dim = 1,
    max_action = 1,
    discount = 0.99,
    memory_size = int(1e6),
    device_name = 'mps',
    batch_size = 1024,
    actor_lr = 1e-4,
    critic_lr = 1e-3,
    tau = 0.05
)

delta = 0.005
log.setup(log_path='training.mcap', level=log.Level.DEBUG)


cartpole = Simulator()
trainer = Trainer(log, config)

total = 0
state = State()
for episode in tqdm.tqdm(count()):
    state = State(cart_position=torch.normal(0.0, 0.2, (1,)).item(), pole_angle=torch.normal(0.0, 0.2, (1, )).item())
    cartpole.reset(state=state)

    for _ in count():
        total += 1
        prev_state = state

        cartpole.advance(delta)
        state = cartpole.get_state()
        reward = compute_reward(state)

        action = trainer.select_action(state, sigma=0.05, train=True, episode=episode, total=total).item()

        trainer.memory.add(prev_state, state, action, reward.value)
        cartpole.set_target(Target(acceleration=action))
        if episode > 5:
            trainer.learn(total=total)

        log.publish('/cartpole/state', state, total)
        log.publish('/cartpole/reward', reward, total)

        if state.error != Error.NO_ERROR:
            break