from time import time
import logging

from core.utils.time_utils import spent_time

logger = logging.getLogger('Training Details')


def training_decorator(func):
    def wrapper(*args, **kwargs):
        t1 = time()
        train_loss, valid_loss = func(*args, **kwargs)
        t2 = time()
        exec_time = spent_time(t1, t2)

        logger.info('\n')
        logger.info(f'   > Train Loss: {train_loss:.3f}')
        logger.info(f'   > Valid Loss: {valid_loss:.3f}')
        logger.info(f'   > Epoch time: {exec_time}\n')

        return train_loss, valid_loss
    return wrapper
