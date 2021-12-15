# Reinforcement-Learning-Inverted-Pendulum

For this project, I really wanted to experiment with reinforcment learning. For this, I am using the OpenAI Gym toolkit to acess a platform for the reinforcment learning enviornemnt. I decided to use reinforcement learning to controll an inverse pendulum because I did a similar problem in ESA, and I love the idea of controlling a pendulum with two drastically different methods.

To controll the inverse pendulum (or as it is called in the Gym toolkit, the cartpole), I am using deep Q-learning. Q-learning is a reinforcement learning algorithm that learns the reward output for a given input. For each new state input, Q-learning returns the optimal action. Deep Q-learning does the same thing, but instead of direclty storing the states and rewards, the algorithm relies on a neural network that connects the inputs to the output (q layer).

In this project, I am using PyTorch to create and optimize the q-learning network. I am creating a network with an input layer that represents the state of the inverse pendulum. In this case, the inputs are position, positional velocity, velocity, and angular velocity. From this layer, the network has two hidden layers with 256 nodes. The last layer contains all the possible Q value actions which for this case is either a force to the right or left on the cart.

The main file of this project allows the user to train the reinforcement learning algorithm for a range of episodes and with a varrying learning rate. I found that in training with a large number of episodes, the model becomes unpredictable and unreliable if the learning rate is set too high. For any training episode count of 300, I recommend setting the learning rate to 0.0001 or lower.

## Installing Prerequisites
This code relies on Pytorch and OpenAI Gym. To install run the commands below.

To install Pytroch:
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
To install OpenAI Gym:
```
pip install gym
```

### To run main program
```
python main.py
```

## Results
Here are a few results of running this algorithm. I ran the reinforcement learning algorithm with episode counts of 10, 20, 50, 100, 200, and 500 with a learning rate of 0.001 and 0.0001.

### Learning rate: 0.001
|![Cart Pendulum Simulation Results](/media/10-games-lr-0.001.gif)10 episodes|![Cart Pendulum Simulation Results](/media/20-games-lr-0.001.gif)20 episodes|![Cart Pendulum Simulation Results](/media/50-games-lr-0.001.gif)50 episodes|
|:-:|:-:|:-:|
|![Cart Pendulum Simulation Results](/media/100-games-lr-0.001.gif)100 episodes|![Cart Pendulum Simulation Results](/media/200-games-lr-0.001.gif)200 episodes|![Cart Pendulum Simulation Results](/media/500-games-lr-0.001.gif)500 episodes|

### Learning rate: 0.0001
|![Cart Pendulum Simulation Results](/media/10-games-lr-0.0001.gif)10 episodes|![Cart Pendulum Simulation Results](/media/20-games-lr-0.0001.gif)20 episodes|![Cart Pendulum Simulation Results](/media/50-games-lr-0.0001.gif)50 episodes|
|:-:|:-:|:-:|
|![Cart Pendulum Simulation Results](/media/100-games-lr-0.0001.gif)100 episodes|![Cart Pendulum Simulation Results](/media/200-games-lr-0.001.gif)200 episodes|![Cart Pendulum Simulation Results](/media/500-games-lr-0.001.gif)500 episodes|


## Some Graphs
### Learning rate: 0.001
|![Cart Pendulum Simulation Results](/media/10-games-lr-0.001.png)10 episodes|![Cart Pendulum Simulation Results](/media/20-games-lr-0.001.png)20 episodes|![Cart Pendulum Simulation Results](/media/50-games-lr-0.001.png)50 episodes|
|:-:|:-:|:-:|
|![Cart Pendulum Simulation Results](/media/100-games-lr-0.001.png)100 episodes|![Cart Pendulum Simulation Results](/media/200-games-lr-0.001.png)200 episodes|![Cart Pendulum Simulation Results](/media/500-games-lr-0.001.png)500 episodes|

### Learning rate: 0.0001
|![Cart Pendulum Simulation Results](/media/10-games-lr-0.0001.png)10 episodes|![Cart Pendulum Simulation Results](/media/20-games-lr-0.0001.png)20 episodes|![Cart Pendulum Simulation Results](/media/50-games-lr-0.0001.png)50 episodes|
|:-:|:-:|:-:|
|![Cart Pendulum Simulation Results](/media/100-games-lr-0.0001.png)100 episodes|![Cart Pendulum Simulation Results](/media/200-games-lr-0.0001.png)200 episodes|![Cart Pendulum Simulation Results](/media/500-games-lr-0.0001.png)500 episodes|


## Resources Used
- https://docs.ray.io/en/latest/rllib-algorithms.html
- https://towardsdatascience.com/ultimate-guide-for-ai-game-creation-part-2-training-e252108dfbd1
- https://www.youtube.com/watch?v=wc-FxNENg9U
- https://medium.com/distributed-computing-with-ray/anatomy-of-a-custom-environment-for-rllib-327157f269e5
- https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
