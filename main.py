import gymnasium as gym
import numpy as np
from Agent import Agent
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve(x, scores, filename):
    """
    Plot della curva dei punteggi medi (rolling average) negli episodi.

    :param x: asse x (di solito numero di episodi)
    :param scores: lista dei punteggi ottenuti a ogni episodio
    :param filename: percorso dove salvare il grafico
    """
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.grid()
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    env = gym.make('Hopper-v5', render_mode='human')
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    print(env.action_space)
    agent = Agent(n_actions=env.action_space.shape[0],input_dims=env.observation_space.shape,batch_size=batch_size,
                  alpha=alpha,n_epochs=n_epochs)
    n_games = 300

    figure_file = 'plot/hopper_learning_curve.png'

    best_score = -np.inf
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation, info = env.reset()
        done = False
        score = 0
        while not done:
            prob, action, val = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            n_steps += 1
            score += reward
            agent.remember(observation,action,prob,val,reward,done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode',i,'score%.1f'%score,'avg score %.1f'%avg_score,'time_stepts',n_steps,'learning_steps',learn_iters)

    env.close()
    
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x,score_history,figure_file)
