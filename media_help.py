from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import csv


def save_frames_as_gif(frames, path='./media/', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def plot_score(scores):
    scores = np.array(scores)
    
    plt.plot(scores)
    plt.xlabel('Epochs')
    plt.ylabel('Game Score')
    plt.title('Training of Reinforcement Learning Model')
    plt.show()

def data2csv(data: list, file: str):
    with open(file,'w') as f:
        for num in data:
            f.write(str(num)+'\n')
