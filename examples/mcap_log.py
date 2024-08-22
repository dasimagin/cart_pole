import cartpole.log as log

from pydantic import BaseModel

import random
import time


# message must be inherited from BaseModel
class RandMsg(BaseModel):
    dist: str = 'uniform(0, 1)'
    value: float = 0.0

mcap = log.MCAPLogger('log_example.mcap', log.DEBUG)

for i in range(10):
    stamp = time.time()
    x = random.uniform(0, 1)

    # publish message, timestamp is optional (default is current time)
    mcap.publish('/random', RandMsg(value=x), stamp)

    # print message to console and log (see /log topic)
    mcap.info(f'publish {x:.2f}', stamp)
