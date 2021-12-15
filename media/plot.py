import numpy as np
import matplotlib.pyplot as plt


def plot_single(games, learning_rate):

    data = np.genfromtxt(f'media/csv/{games}-games-lr-{learning_rate}.csv')

    plt.plot(data)
    plt.xlabel('Episodes')
    plt.ylabel('Training Score')
    plt.legend([f'Learning Rate = {learning_rate}'])
    #plt.show()
    plt.savefig(f'media/{games}-games-lr-{learning_rate}.png')


plot_single(500,0.001)
