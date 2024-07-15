from cartpole import log
from cartpole.device import CartPoleDevice
from cartpole.common import Target
import time


def main():
    log.setup(log_path="untracked/device_test.mcap")
    logger = log.get_logger()
    device = CartPoleDevice(hard_reset=True)

    limit = 5  # seconds
    start = time.time()
    while time.time() - start < limit:
        state = device.get_state()
        logger.publish("/cartpole/state", state)


if __name__ == "__main__":
    from logging import basicConfig
    basicConfig(level="INFO")
    main()
