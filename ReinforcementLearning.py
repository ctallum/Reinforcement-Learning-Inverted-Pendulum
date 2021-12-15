import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, learning_rate, input_dims, layer1_dims, layer2_dims, q_actions):
        super(DeepQNetwork, self).__init__()
        # input layer
        self.input_dims = input_dims
        # hidden layer 1
        self.layer1_dims= layer1_dims
        # hidden layer 2
        self.layer2_dims = layer2_dims
        # output layer
        self.q_actions = q_actions
        # create nn layers
        self.layer1 = nn.Linear(*self.input_dims, self.layer1_dims)
        self.layer2= nn.Linear(self.layer1_dims, self.layer2_dims) 
        self.q_layer = nn.Linear(self.layer2_dims, self.q_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        # slap some GPU on there if possible
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # forward pass, 2 relu layer
    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        actions = self.q_layer(x)

        return actions

class Agent():
    def __init__(self, gamma, epsilon, learning_rate, input_dims, batch_size, q_actions, max_mem_size=10000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma # discount factor, 0-1, importance of future moves
        self.epsilon = epsilon # how often algorithm takes random versus planned path
        self.eps_min = eps_end # lowest epsilon value
        self.eps_dec = eps_dec # as learning progresses, epsilon decreases
        self.learning_rate = learning_rate
        self.action_space = [i for i in range(q_actions)]
        self.mem_size = max_mem_size # max size of arrays to store q-learning data
        self.batch_size = batch_size
        self.mem_counter = 0

        # setup learning network
        self.Q_eval = DeepQNetwork(self.learning_rate, input_dims=input_dims, layer1_dims=256, layer2_dims=256, q_actions=q_actions)

        # setup memory management as numpy arrays
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        # wrap index around if it goes past end of array
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def choose_action(self, observation):
        # if weight of random is greater than epsilon, calculate best option
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            # forward pass
            actions = self.Q_eval.forward(state)
            # find maximum value of forward pass
            action = T.argmax(actions).item()
        else:
            # take random action
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        # don't both learning if batch size memory counter is smaller than a batch
        if self.mem_counter < self.batch_size:
            return
        
        # do gradient descent
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        # backwards pass to optimize weights
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        # decrease epsilon by decriment unless already at lowest val
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min


    