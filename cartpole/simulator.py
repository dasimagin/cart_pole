from cartpole.common import (
    CartPoleBase,
    Config,
    Error,
    Limits,
    Parameters,
    State,
    Target,
)

from pydantic import BaseModel
from typing import Any

import numpy


def sgn(x: float) -> int:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def snap_zero(x: float, eps: float) -> float:
    return 0 if abs(x) < eps else x


class SimulatorInfo(BaseModel):
    """
    Simulator info.

    Attributes
    ----------
    step_count: int
        number of simulation steps
    integration_count: int
        number of integration steps
    """

    step_count: int = 0
    integration_count: int = 0


class Simulator(CartPoleBase):
    """
    CartPole simulator (integration implemented using RK4 method).

    For more information see: https://cartpole.robotics-lab.ru/dynamics-and-control
    """

    def __init__(self, integration_step: float = 0.001):
        """
        Create simulator with some default config.

        Parameters
        ----------
        integration_step: float
            integration time step (s)
        """

        # set default config
        self._config = Config(
            hardware_limit=Limits(
                cart_position=0.5, cart_velocity=2.0, cart_acceleration=5.0
            ),
            control_limit=Limits(),
            parameters=Parameters(g=9.81, b=0, k=0.3),
        )

        self._state = State(error=Error.NEED_RESET)
        self._target = Target(velocity=0)

        self._integration_step = integration_step
        self._integration_count = 0
        self._step_count = 0

    def get_config(self) -> Config:
        """
        Return current config.
        """
        return self._config

    def set_config(self, config: Config) -> None:
        """
        Set new config. In case of invalid config, AssertionError is raised.
        """

        assert config.control_limit.stronger(config.hardware_limit)

        assert config.parameters.friction_coef is not None
        assert config.parameters.friction_coef >= 0

        assert config.parameters.mass_coef is not None
        assert config.parameters.mass_coef > 0

        self._config = config

    def _eval_acceleration_by_velocity(
        self, velocity: float, eps: float = 1e-6
    ) -> float:
        err = velocity - self._state.cart_velocity

        a = self._target.acceleration_or(self._config.control_limit.cart_acceleration)
        if abs(err) < a * self._integration_step:
            return snap_zero(err / self._integration_step, eps)

        return sgn(err) * a

    def _eval_cart_acceleration(self, eps: float = 1e-6) -> float:
        if self._target.position is not None:
            a = self._target.acceleration_or(
                self._config.control_limit.cart_acceleration
            )
            v = self._target.velocity_or(self._config.control_limit.cart_velocity)
            err = self._target.position - self._state.cart_position

            # print(f'_eval_cart_acceleration err={err}, v={v}, a={a}, state_v={self._state.cart_velocity}')
            if abs(err) < 1e-3:
                return self._eval_acceleration_by_velocity(0)
            if sgn(err) != sgn(self._state.cart_velocity):
                # stoppping to change direction
                return self._eval_acceleration_by_velocity(sgn(err) * v)
            else:
                # bang-bang strategy
                a_brake = self._state.cart_velocity**2 / (2 * abs(err))
                # print(f'a_brake = {a_brake}, a = {a}')
                if a_brake >= a:
                    return self._eval_acceleration_by_velocity(0)
                else:
                    return self._eval_acceleration_by_velocity(sgn(err) * v)

        if self._target.velocity is not None:
            return self._eval_acceleration_by_velocity(self._target.velocity, eps)

        if self._target.acceleration is not None:
            return self._target.acceleration

        raise Exception("At least one of the target is required")

    def _derivative(self, s: numpy.ndarray, a: float) -> numpy.ndarray:
        """
        Calculate derivative of the given state (used for integration).

        Parameters
        ----------
        s: numpy.ndarray
            state vector
        a: float
            cart acceleration (m/s^2)

        Returns
        -------
        :numpy.ndarray
            derivative of the given state
        """

        result = numpy.zeros(4)

        b = self._config.parameters.friction_coef
        k = self._config.parameters.mass_coef
        g = self._config.parameters.gravity

        result[0] = s[2]
        result[1] = s[3]
        result[2] = a
        result[3] = -b * s[3] - k * (a * numpy.cos(s[1]) + g * numpy.sin(s[1]))

        return result

    def reset(self, state: State = State()) -> None:
        """
        Reset simulator to the given state.
        """

        self._state = state
        self._target = Target(velocity=0)

        self._count = 0
        self._integration_count = 0

    def get_state(self) -> State:
        """
        Return current state.
        """

        return self._state

    def get_info(self) -> SimulatorInfo:
        """
        Return current simulator info.
        """

        return SimulatorInfo(
            step_count=self._step_count, integration_count=self._integration_count
        )

    def set_target(self, target: Target) -> State:
        """
        Set control target and return current state.
        """

        self._target = target

        if not self._state.error:
            self._state.cart_acceleration = self._eval_cart_acceleration()
            self._state.validate(self._config)

        return self.get_state()

    def advance(self, delta: float) -> None:
        """
        Make simulation step of the given length.
        For integration, RK4 method is used.

        Parameters
        ----------
        delta: float
            length of the simulation step (s)
        """

        if self._state.error:
            return

        s = self._state.numpy4()
        h = self._integration_step
        h_2 = h / 2

        integration_step_n = int(delta / h)

        for _ in range(integration_step_n):
            a = self._eval_cart_acceleration()

            k1 = self._derivative(s, a)
            k2 = self._derivative(s + k1 * h_2, a)
            k3 = self._derivative(s + k2 * h_2, a)
            k4 = self._derivative(s + k3 * h, a)

            s += (k1 + 2 * k2 + 2 * k3 + k4) * h / 6

            self._integration_count += 1

            self._state = State(
                cart_position=s[0],
                cart_velocity=s[2],
                cart_acceleration=a,
                pole_angle=s[1],
                pole_angular_velocity=s[3],
                stamp=self._integration_step * self._integration_count,
            )

            self._state.validate(self._config)

        self._step_count += 1

    def close(self) -> None:
        """
        Clear any resources used by the simulator.
        """
        pass
