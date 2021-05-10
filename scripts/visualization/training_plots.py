from typing import List

import matplotlib.pyplot as plt


def plot_training_step(train_losses: List,
                       valid_losses: List):

    plt.title('Losses')
    plt.plot(train_losses, c='r', label='Training Loss')
    plt.plot(valid_losses, c='b', label='Validation Loss')
    plt.grid()
    plt.legend()
    plt.show()


