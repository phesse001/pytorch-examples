import gym
from dqn import Agent
import numpy as np
import torch

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 64,
            n_actions = 4, eps_end = 0.01, input_dims = [8], alpha = 0.001)
    scores, eps_history = [], []
    n_games = 500
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            new_observation, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(action, observation, reward, new_observation, done)
            agent.learn()
            observation = new_observation
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        print("score " + str(score) +  "|average score " + str(avg_score) + "|epsilon " + str(agent.epsilon))

    #save the model
    PATH = './dqn.pth'
    torch.save(agent.Q_eval.state_dict(), PATH)