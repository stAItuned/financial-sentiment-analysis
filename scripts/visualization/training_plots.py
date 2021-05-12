from typing import List, Text

import matplotlib.pyplot as plt


def plot_training_step(train_losses: List,
                       valid_losses: List,
                       save_dir: Text = None,
                       timestamp: Text = None):

    plt.title('Losses')
    plt.plot(train_losses, c='r', label='Training Loss')
    plt.plot(valid_losses, c='b', label='Validation Loss')
    plt.grid()
    plt.legend()

    if save_dir and timestamp:
        plot_path = f'{save_dir}training_plot_{timestamp}.png'
        plt.savefig(plot_path)

    plt.show()


