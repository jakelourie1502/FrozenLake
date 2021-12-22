import numpy as np
from frozen_lake import gridWorld
from Sarsa import *
np.set_printoptions(suppress=True, precision=3)
from utils import print_state_vals, print_policy

# Get Environment #
size = (4,4)
lakes = [(1,1),(1,3),(2,3),(3,0)]
goals = {(3,3):1}
dist = np.zeros((size[0]*size[1]+1))
dist[0]=1
env=gridWorld(size,lakes,goals, n_actions = 4, max_steps = 100, dist = dist, seed = None, rnd=0.1)
print(env.goal_states_idx)
print(env.board)

# Sarsa #
print('##_Sarsa')
policy, value, episodes= sarsa(env,max_episodes=2000,eta=0.5,gamma=0.9,epsilon=1,seed=0)
print('Number of episodes:',episodes)
print_state_vals(env, value)
print_policy(env, policy)
#env.render(policy,value)

# Sarsa #
# print('##_Q-learning')
# policy, value, episodes = sarsa(env,max_episodes=5000,eta=0.5,gamma=0.9,epsilon=0.5,seed=0)
# print(policy)
# print(value)
# print('Number of episodes:',episodes)
# #env.render(policy,value)