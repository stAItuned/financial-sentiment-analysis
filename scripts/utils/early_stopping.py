

class EarlyStopping:

    def __init__(self,
                 patience: int,
                 min_delta: float = 0):

        self.patience = patience
        self.min_delta = min_delta
        self.cum_patience = 0

    def check_stopping(self,
                       loss: float,
                       previous_loss: float):

        if loss > previous_loss and abs(loss - previous_loss) > self.min_delta:
            self.cum_patience += 1

        return self.cum_patience < self.patience

