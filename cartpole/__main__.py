from cartpole.common import Config, Limits, Parameters
from cartpole.common import CartPoleBase, Error, State, Target
from cartpole.simulator import Simulator
from cartpole.eval import find_parameters

import cartpole.log as log

import argparse
import random
import time


def parse_args():
    common = argparse.ArgumentParser(
        prog='cartpole',
        description='cartpole control experiments'
    )

    subparsers = common.add_subparsers(title='commands', dest='command', required=True, help='command help')

    # common arguments

    common.add_argument('-S', '--simulation', action='store_true', help='simulation mode')
    common.add_argument('-c', '--config', type=str, help='cartpole yaml config file')
    common.add_argument('-m', '--mcap', type=str, default='', help='mcap log file')
    common.add_argument('-a', '--advance', type=float, default=0.01, help='advance simulation time (seconds)')

    # eval arguments
    eval = subparsers.add_parser('eval', help='system identification')

    eval.add_argument('-d', '--duration', type=float, default=10.0, help='experiment duration (seconds)')
    eval.add_argument('-O', '--output', type=str, help='output yaml config file')

    return common.parse_args()


def evaluate(device: CartPoleBase, config: Config, args: argparse.Namespace) -> None:
    log.info('parameters evaluation')

    position_margin = 0.01
    position_tolerance = 0.005
    duration = args.duration
    advance = args.advance

    position_max = config.control_limit.cart_position - position_margin
    velocity_max = config.control_limit.cart_velocity
    acceleration_max = config.control_limit.cart_acceleration

    log.info(f'run calibration session for {duration:.2f} seconds')
    device.reset()

    start = device.get_state()
    state = start

    target = Target(position=0, velocity=0, acceleration=0)
    states = []

    while not state.error and state.stamp - start.stamp < duration:
        state = device.get_state()

        if abs(target.position - state.cart_position) < position_tolerance:
            position = random.uniform(position_max/2, position_max)
            target.position = position if target.position < 0 else -position
            target.velocity = random.uniform(velocity_max/2, velocity_max)
            target.acceleration = random.uniform(acceleration_max/2, acceleration_max)

            log.info(f'target {target}')
            device.set_target(target)

        log.publish('/cartpole/state', state, state.stamp)
        log.publish('/cartpole/target', target, state.stamp)
        log.publish('/cartpole/info', device.get_info(), state.stamp)

        states.append(state)
        device.advance(advance)

        if args.simulation:
            time.sleep(advance) # simulate real time

    log.info(f'find parameters')
    parameters = find_parameters(states, config.parameters.gravity)

    log.info(f'parameters: {parameters}')

    if args.output:
        log.debug(f'save config to {args.output}')

        config = config.copy(deep=True)
        config.parameters = parameters

        config.to_yaml_file(args.output)


def main():
    args = parse_args()

    log.setup(log_path=args.mcap, level=log.Level.DEBUG)
    # TODO(@dasimagin) fix this dirty hack
    # wait for foxglove server to start
    time.sleep(5)

    if args.mcap:
        log.debug(f'mcap file: {args.mcap}')
    else:
        log.debug('no mcap file specified')

    if args.simulation:
        log.info('simulation mode')
        device = Simulator(integration_step=min(args.advance/20, 0.001))
    else:
        raise NotImplementedError()
    
    if args.config:
        log.debug(f'config file: {args.config}')
        config = Config.from_yaml_file(args.config)
    else:
        log.warning('no config file specified, using defaults')
        config = Config()


    device.set_config(config)

    if args.command == 'eval':
        evaluate(device, config, args)
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    main()
