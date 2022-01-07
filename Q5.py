import numpy as np
from frozen_lake import gridWorld
from Sarsa import *
np.set_printoptions(suppress=True, precision=3)
from utils import check_optimal,prep_optimal
from Q_Learning import *
from Default_Lakes import *
from policy_iteration import *


##Big Lake###
print('Big Lake')

# Get Environment #
size, lakes, goals, dist = big_lake()
env=gridWorld(size,lakes,goals, n_actions = 4, max_steps = 100, dist = dist, seed = None, rnd=0.1)
print(env.board)

# Get optimal policy #
gamma = 0.9
theta = 0.001
max_iterations = 100
print('## Policy iteration')
policy, value, it_counter = policy_iteration(env, gamma, theta, max_iterations)
optimal_policy = prep_optimal(env,policy)
env.render(policy, value)

# Sarsa #
print('##_Sarsa')
policy, value, episodes= sarsa(env,max_episodes=500000,eta=0.2,gamma=0.9,epsilon=0.8,optimal_policy=optimal_policy,seed=0,initial_q=1,
                               eta_floor = 0.001,epsilon_floor=0.25,epsilon_ramp_epoch=40000,eta_ramp_epoch=300000,madness=10)
print('Number of episodes', episodes)
env.render(policy,value)

# Q Learning #
print('##_Q-learning')
policy, value, episodes = q_learning(env,max_episodes=500000,eta=0.2,gamma=0.9,epsilon=0.8,optimal_policy=optimal_policy,seed=0,initial_q=1,
                               eta_floor = 0.001,epsilon_floor=0.25,epsilon_ramp_epoch=40000,eta_ramp_epoch=300000,madness=10)
print('Number of episodes:',episodes)
env.render(policy,value)
