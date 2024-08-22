import cartpole.log as log

from pydantic import BaseModel

import random
import time


# message must be inherited from BaseModel
class RandMsg(BaseModel):
    dist: str = 'uniform(0, 1)'
    value: float = 0.0

foxglove = log.FoxgloveWebsocketLogger(log.DEBUG)

for i in range(100):
    stamp = time.time()
    x = random.uniform(0, 1)

    # publish message, timestamp is optional (default is current time)
    foxglove.publish('/random', RandMsg(value=x), stamp)

    # print message to console and log (see /log topic)
    foxglove.info(f'publish {x:.2f}', stamp)

    time.sleep(0.2)
