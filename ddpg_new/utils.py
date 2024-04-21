from cartpole.common import State, BaseModel, Error
from math import pi, cos, tanh
import torch

class Scalar(BaseModel):
    value: float

class Config(BaseModel):
    state_dim: int
    action_dim: int
    memory_size: int
    device_name: str 
    batch_size: int
    actor_lr: float
    actor_width: int
    actor_grad_norm: float
    critic_lr: float
    critic_width: int
    critic_grad_norm: float
    critic_noise: float
    tau: float
    max_action: float
    finish_penalty: float = -1
    discount: float = 0.99
    max_position: float = 0.5
    max_velocity: float = 2.0


def state_to_tensor(state: State, config: Config):
    return torch.tensor([
        state.cart_position / config.max_position,
        state.cart_velocity / config.max_velocity,
        state.cart_acceleration / config.max_action,
        1-cos(state.pole_angle),
        tanh(state.pole_angular_velocity),
    ])

def make_tensor(from_state, to_state, action, reward, config: Config, done: bool):
    return torch.concat([
        state_to_tensor(from_state, config),
        state_to_tensor(to_state, config),
        torch.tensor([action]),
        torch.tensor([reward]),
        torch.tensor([done])
    ])