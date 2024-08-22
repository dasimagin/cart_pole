from cartpole.common import Actor, Config, Limits, Parameters, State, Target
from pydantic import BaseModel
from scipy.linalg import solve_continuous_are

import numpy as np


def clamp(x, min_x, max_x):
    return min(max_x, max(min_x, x))

class LQRBalancerParams(BaseModel):
    '''
    LQR parameters.

    Q matrix:
        * cart_position_cost
        * pole_angle_cost
        * cart_velocity_cost
        * pole_angular_velocity_cost
    
    R matrix:
        * cart_acceleration_cost
    '''

    cart_position_cost: float = 0.0
    cart_velocity_cost: float = 0.0
    cart_acceleration_cost: float = 0.0

    pole_angle_cost: float = 0.0
    pole_angular_velocity_cost: float = 0.0
    
class LQRBalancer(Actor):
    '''
    LQR-based pole balancer actor at the unstable equilibrium point.

    Equlibrium point:
        * cart_position = 0
        * pole_angle = PI
        * cart_velocity = 0
        * pole_angular_velocity = 0

    For more information see:
        https://en.wikipedia.org/wiki/Linear-quadratic_regulator
    '''
        
    def __init__(self, config: Config, lqr_params: LQRBalancerParams):
        self._config = config
        self._lqr_params = lqr_params

        k = config.parameters.mass_coef
        b = config.parameters.friction_coef
        g = config.parameters.gravity

        A = np.array([
            [0,   0, 1,  0],
            [0,   0, 0,  1],
            [0,   0, 0,  0],
            [0, k*g, 0, -b],
        ])
    
        B = np.array([[0], [0], [1], [k]])
    
        Q = np.diag([
            params.cart_position_cost,
            params.pole_angle_cost,
            params.cart_velocity_cost,
            params.pole_angular_velocity_cost,
        ])
    
        R = np.array([
            [params.cart_acceleration_cost]
        ])

        P = solve_continuous_are(A, B, Q, R)
        
        self.control = -np.linalg.inv(R) @ B.T @ P
        self.target = State(
            cart_position=0,
            cart_velocity=0,
            pole_angle=np.pi,
            pole_angular_velocity=0,
        )

    def __call__(self, state: State) -> Target:
        '''
        Calculate control action for the given state.
        '''

        error = state.numpy4() - taget.numpy4()

        acceleration = clamp(
            self.control @ error,
            -self._config.control_limit.cart_acceleration,
            +self._config.control_limit.cart_acceleration)

        return Target(acceleration=acceleration)
