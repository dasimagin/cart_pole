from cartpole.common import State, BaseModel, Error
from math import pi, cos, sin, tanh
import torch

class Scalar(BaseModel):
    value: float

class ExperimentConfig(BaseModel):
    action_noise: float
    warmup_stay: int
    warmup_random: int
    actor_start: int
    critic_start: int
    state_dim: int
    action_dim: int
    memory_size: int
    device_name: str
    batch_size: int
    actor_lr: float
    actor_width: int
    actor_layers: int
    actor_grad_norm: float
    critic_lr: float
    critic_width: int
    critic_layers: int
    critic_grad_norm: float
    critic_noise: float
    tau: float
    max_action: float
    finish_penalty: float = -1
    discount: float = 0.99
    max_position: float = 0.5
    max_velocity: float = 2.0
    episodes_count: int
    delta: float


def state_to_tensor(state: State, config: ExperimentConfig):
    return torch.tensor([
        state.cart_position / config.max_position,
        state.cart_velocity / config.max_velocity,
        cos(state.pole_angle),
        sin(state.pole_angle),
        state.pole_angle/(4*pi),
        state.pole_angular_velocity/(4*pi)
    ])

def make_tensor(from_state, to_state, action, reward, config: ExperimentConfig, done: bool):
    return torch.concat([
        state_to_tensor(from_state, config),
        state_to_tensor(to_state, config),
        torch.tensor([action]),
        torch.tensor([reward]),
        torch.tensor([done])
    ])
