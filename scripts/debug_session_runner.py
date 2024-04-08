import logging
import time
from pathlib import Path

from cartpole.actors.demo import DemoActor
from cartpole.common.interface import CartPoleBase, Config, Error
from cartpole.common.util import init_logging
from cartpole.device import CartPoleDevice
from cartpole.sessions.actor import Actor
from cartpole.sessions.collector import CollectorProxy

LOGGER = logging.getLogger("debug-session-runner")


def control_loop(device: CartPoleBase, actor: Actor, max_duration: float):
    max_accel = 6
    start = time.perf_counter()
    while time.perf_counter() - start < max_duration:
        state = device.get_state()
        if state.error != Error.NO_ERROR:
            break
        # print("ANGLE:", state.pole_angle)
        stamp = time.perf_counter() - start
        target = actor(state, stamp=stamp)
        target = min(max_accel, max(-max_accel, target))
        # logging.info("STAMP: %s", stamp)
        device.set_target(target)
        device.advance(0)


def reset_pole_angle(device: CartPoleDevice):
    # FIXME: Move to firmware
    device.rotations = 0
    device.prev_angle = 0
    device.zero_angle = device.get_state().pole_angle
    LOGGER.info(f"Pole angle correction: {device.zero_angle:.5f}")


def wait_pole_stabilization(device: CartPoleDevice):
    LOGGER.info("Waiting for pole stabilization...")
    last_time = time.perf_counter()
    while True:
        state = device.get_state()
        # print(f"{state.pole_angular_velocity:.2f}")
        if abs(state.pole_angular_velocity) > 0.1:
            last_time = time.perf_counter()
        if time.perf_counter() - last_time > 3:
            LOGGER.info("Pole stabilized")
            break
    time.sleep(1)


if __name__ == "__main__":
    from cartpole.common.util import init_logging

    init_logging()

    SESSION_ID = "demo-session"
    SESSION_MAX_DURATION = 60  # Seconds
    OUTPUT_PATH = Path(f"data/sessions/{SESSION_ID}")

    ACTOR_CLASS = DemoActor
    DEVICE_CONFIG = Config(
        max_position=0.26,
        max_velocity=5,
        max_acceleration=10.0,
        clamp_velocity=True,
        clamp_acceleration=True,
    )
    ACTOR_CONFIG = dict(
        # config=Config(max_position=0.22, max_velocity=3, max_acceleration=5.0, pole_length=0.16)
        config=Config(max_position=0.25, max_velocity=2, max_acceleration=4, pole_length=0.18)
    )

    device = CartPoleDevice()
    proxy = CollectorProxy(
        cart_pole=device,
        actor_class=ACTOR_CLASS,
        actor_config=ACTOR_CONFIG,
    )

    try:
        proxy.reset(DEVICE_CONFIG)
        actor = DemoActor(**ACTOR_CONFIG)
        # raise RuntimeError
        actor.proxy = proxy
        wait_pole_stabilization(device)
        reset_pole_angle(device)  # FIXME
        control_loop(proxy, actor, max_duration=SESSION_MAX_DURATION)
    except Exception:
        LOGGER.exception("Aborting run due to error")
    finally:
        LOGGER.info("Run finished")
        proxy.set_target(0)
        proxy.close()

    proxy.save(OUTPUT_PATH / "session.json")
