import gym
from ReinforcementLearning import Agent
import numpy as np
import time
from tqdm import tqdm
import os
clear = lambda: os.system('clear')
    

from media_help import *

def train_agent(env, n_games, learning_rate):
    # parameters
    gamma = 0.99 # discount rate
    epsilon = 1.0 
    eps_end = 0.01
    batch_size = 64
    #learning_rate = 0.0001
    q_actions = env.action_space.n # figure out how many inputs to env there are
    input_dims = [len(env.observation_space.high)]
    agent = Agent(gamma=gamma, epsilon=epsilon, batch_size=batch_size, q_actions=q_actions, eps_end=eps_end, input_dims=input_dims, learning_rate=learning_rate)
    scores, eps_history = [], []
    
    # learning episodes
    

    for game in tqdm(range(n_games), ascii= "---------#", desc = 'Training Algorithm'):
        if game == 1:
            clear()
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)

            agent.learn()
            observation = observation_

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        
        '''
        print('episode', game, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
        '''
    return agent, scores

if __name__ == "__main__":

    env = gym.make('CartPole-v1')

    generate_csv = False
    generate_gif = False


    # print('Running environment with no training')
    # time.sleep(2)
    # done = False
    # env.reset()
    # while not done:
    #     time.sleep(0.01)
    #     env.render()
    #     _, _, done, _= env.step(env.action_space.sample()) # take a random action
    # time.sleep(1)
    # env.close()
    
    # run mass iteration with different episodes
    episodes = [10, 20]
    for n_games in episodes:

        # set learning rate 
        lr = .0001
        agent, scores = train_agent(env, n_games, lr)

        print('Running environment after training')
        time.sleep(2)
        done = False
        observation = env.reset()
        frames = []
        while not done:
            time.sleep(.01)
            frames.append(env.render(mode="rgb_array"))
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward, observation_, done)
            agent.learn()
            observation = observation_
        env.close()
        
        #plot_score(scores)
        if generate_csv:
            data2csv(scores,f'./media/{n_games}-games-lr-{lr}.csv')
        if generate_gif:
            save_frames_as_gif(frames,filename=f'{n_games}-games-lr-{lr}.gif')
