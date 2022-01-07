from frozen_lake import *
from policy_iteration import *
from value_iteration import *
from Sarsa import *
from Q_Learning import *
from linear_approximation import *
from Default_Lakes import *
from utils import *


################ Main function ################

def main():
    seed = 0
    
    # Small lake
    lake =   [['&', '.', '.', '.'],
              ['.', '#', '.', '#'],
              ['.', '.', '.', '#'],
              ['#', '.', '.', '$']]

    size, lakes, goals, dist = small_lake()
    env = gridWorld(size,lakes,goals, n_actions = 4, max_steps = 100, dist = dist, seed = None, rnd=0.1)
    
    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100
    
    print('')

    print('## Policy iteration')
    policy, value, it_counter = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)
    # Store Optimal policy for Sarsa and Q-Learning #
    optimal_policy = prep_optimal(env,policy)
    print('')

    print('## Value iteration')
    policy, value, it_counter = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')
    
    print('# Model-free algorithms')
    max_episodes = 10000
    eta = 0.5
    epsilon = 0.8
    
    print('')

    print('##_Sarsa')
    policy, value, episodes = sarsa(env, max_episodes=max_episodes, eta=eta, gamma=gamma, epsilon=epsilon,
                                     seed=seed)
    print('Number of episodes', episodes)
    env.render(policy, value)
    
    print('')

    print('##_Q-learning')
    policy, value, episodes = q_learning(env, max_episodes=max_episodes, eta=eta, gamma=gamma, epsilon=epsilon,
                                          seed=seed)
    print('Number of episodes:', episodes)
    env.render(policy, value)
    
    print('')

    linear_env = LinearWrapper(env)

    print('## Linear Sarsa')

    parameters = linear_sarsa(linear_env, max_episodes, eta,
                              gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')

    print('## Linear Q-learning')

    parameters = linear_q_learning(linear_env, max_episodes, eta,
                                   gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

main()