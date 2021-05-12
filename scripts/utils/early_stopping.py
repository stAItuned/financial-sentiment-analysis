
import logging

logger = logging.getLogger('Early Stopping')


class EarlyStopping:

    def __init__(self,
                 patience: int,
                 min_delta: float = 0):

        self.patience = patience
        self.min_delta = 0 if min_delta is None else min_delta
        self.cum_patience = 0

    def check_stopping(self,
                       loss: float,
                       previous_loss: float):

        if loss > previous_loss and abs(loss - previous_loss) > self.min_delta:
            self.cum_patience += 1
            logger.info(f' > Updating patience: {self.cum_patience}')

        return self.cum_patience < self.patience

