from frozen_lake import *
from policy_iteration import *
from value_iteration import *
from Sarsa import *
from Q_Learning import *
from linear_approximation import *
from Default_Lakes import *


################ Main function ################

def main():
    seed = 0
    
    # Small lake
    lake =   [['&', '.', '.', '.'],
              ['.', '#', '.', '#'],
              ['.', '.', '.', '#'],
              ['#', '.', '.', '$']]

    size, lakes, goals, dist = big_lake()
    env = gridWorld(size,lakes,goals, n_actions = 4, max_steps = 100, dist = dist, seed = None, rnd=0.1)
    
    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100
    
    print('')

    print('## Policy iteration')
    policy, value, it_counter = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('## Value iteration')
    policy, value, it_counter = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')
    
    print('# Model-free algorithms')
    max_episodes = 2000
    eta = 0.5
    epsilon = 1
    
    print('')
    
    print('## Sarsa')
    policy, value,_ = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed)
    env.render(policy, value)
    
    print('')
    
    print('## Q-learning')
    policy, value,_ = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed)
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