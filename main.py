import gym
from ReinforcementLearning import Agent
import numpy as np

if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[8], learning_rate=0.003)
    scores, eps_history = [], []
    n_games = 50

    observation = env.reset()
    for _ in range(1000):
        env.render()
        action = agent.choose_action(observation)
        observation, reward, done, info = env.step(action) 
        if done:
            break
    env.close()

    for game in range(n_games):
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

        print('episode', game, 'score %.2f' % score,
                'average score %.2f' % avg_score,
                'epsilon %.2f' % agent.epsilon)
    
    observation = env.reset()
    for _ in range(10000):
        env.render()
        action = agent.choose_action(observation)
        observation, reward, done, info = env.step(action) 
        if done:
            break
    env.close()