from time import time
import logging

from core.utils.time_utils import spent_time

logger = logging.getLogger()


def timing(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        logger.info(f' >>  {func.__name__}: Starts')
        result = func(*args, **kwargs)
        t2 = time()
        exec_time = spent_time(t1, t2)
        logger.info(f' >>  {func.__name__}: Executed in {exec_time}')
        return result
    return wrapper
