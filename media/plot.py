import numpy as np
import matplotlib.pyplot as plt


def plot_single(games, learning_rate):

    data = np.genfromtxt(f'media/csv/{games}-games-lr-{learning_rate}.csv')
    plt.figure()
    plt.plot(data)
    plt.xlabel('Episodes')
    plt.ylabel('Training Score')
    plt.legend([f'Learning Rate = {learning_rate}'])
    plt.title(f'Game results with {games} episode trainings and {learning_rate} learning rate')
    #plt.show()
    plt.savefig(f'media/{games}-games-lr-{learning_rate}.png')


for episode in [10, 20, 50, 100, 200, 500]:
    for lr in [0.001, 0.0001]:
        plot_single(episode,lr)
