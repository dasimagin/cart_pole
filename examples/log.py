
import cartpole.log as log

from pydantic import BaseModel

import random
import time

# all messages must be inherited from BaseModel
class RandMsg(BaseModel):
    dist: str = 'uniform(0, 1)'
    value: float = 0.0

# define log file name
log.setup(log_path='log_example.mcap', level=log.DEBUG)

# messages are available in real time in foxglove (websocket mode)
for i in range(10):
    value = random.uniform(0, 1)

    # publish message, timestamp is optional (default is current time)
    log.publish('/random', RandMsg(value=value))

    # print message to console and log (see /log topic)
    log.info(f'publish {value:.2f}')

    # add some delay for demo purposes
    time.sleep(0.2)


