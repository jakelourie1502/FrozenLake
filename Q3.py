import numpy as np
from frozen_lake import gridWorld
from Sarsa import *
np.set_printoptions(suppress=True, precision=3)
from utils import check_optimal,prep_optimal
from Q_Learning import *
from Default_Lakes import *
from policy_iteration import *

###Small Lake###
print('## Small Lake')

# Get Environment #
size, lakes, goals, dist = small_lake()
env=gridWorld(size,lakes,goals, n_actions = 4, max_steps = 100, dist = dist, seed = None, rnd=0.1)
print(env.board)

gamma = 0.9
theta = 0.001
max_iterations = 100
print('## Policy iteration')
policy, value, it_counter = policy_iteration(env, gamma, theta, max_iterations)
optimal_policy = prep_optimal(env,policy)
env.render(policy, value)

# Sarsa #
print('##_Sarsa')
policy, value, episodes= sarsa(env,max_episodes=10000,eta=0.5,gamma=0.9,epsilon=1,optimal_policy=optimal_policy,seed=0)
print('Number of episodes',episodes)
env.render(policy,value)

# # Q Learning #
print('##_Q-learning')
policy, value, episodes = q_learning(env,max_episodes=10000,eta=0.5,gamma=0.9,epsilon=1,optimal_policy=optimal_policy,seed=0)
print('Number of episodes:',episodes)
env.render(policy,value)