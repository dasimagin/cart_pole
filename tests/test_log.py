from cartpole import log, State
from pydantic import BaseModel
from time import sleep, time

import pytest

class SomeObject(BaseModel):
    name: str = ''
    value: float = 0.0

def test_log_some_object():
    # use default setting and lazy initialization

    # send some object, use current time
    log.publish('/some/object', SomeObject(name='first', value=1.0))

    # send another object, specify time explicitly
    log.publish('/some/object', SomeObject(name='second', value=2.0), time())


def test_log_state():
    # explicitly setup logger, specify log path
    log.setup(log_path='test.mcap')

    # send state
    log.publish('/cartpole/state', State())


def test_log_bad_object():
    # explicitly setup logger only for foxglove
    log.setup()

    # send bad object (not pydantic model)
    log.publish('/str', 'something')

    # after a while get an forwarded error from foxglove thread
    sleep(0.1)
    with pytest.raises(AssertionError):
        log.publish('/some/object', SomeObject())

def test_log_different_types_to_one_topic():
    # reset logger, after fail before and setup for mcap
    log.setup('test.mcap')
    topic = '/cartpole/state'

    log.publish(topic, State())

    # send another type to the same topic, now we get an error immediately,
    # because mcap logger works at the same thread
    with pytest.raises(AssertionError):
        log.publish(topic, SomeObject())
