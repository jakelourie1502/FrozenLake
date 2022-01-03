from frozen_lake import gridWorld as FrozenLake
from linear_approximation import LinearWrapper, linear_sarsa, linear_q_learning
import numpy as np
from utils import print_state_vals, print_policy, small_lake

# ---------- Load in the environment ------------
size = (4, 4)
lakes = [(1, 1), (1, 3), (2, 3), (3, 0)]
goals = {(3, 3): 1}
dist = np.zeros((size[0] * size[1] + 1))
dist[0] = 1
env = FrozenLake(size, lakes, goals, n_actions=4, max_steps=100, dist=dist, seed=None, rnd=0.1)
state = env.reset()
env.render()
actions = [0, 1, 2, 3]
done = False

# --------- Linear Approximation parameters ---------
max_episodes = 2000
eta = 0.5
epsilon = 0.5
gamma = 0.9
seed = 0
linear_env = LinearWrapper(env)
parameters_sarsa = linear_sarsa(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)
parameters_q_learning = linear_q_learning(linear_env, max_episodes, eta, gamma, epsilon, seed=seed)

policy_sarsa, value_sarsa = linear_env.decode_policy(parameters_sarsa)
policy_q_learning, value_q_learning = linear_env.decode_policy(parameters_q_learning)
print(value_q_learning)

print(value_sarsa)

# FIXME: Render doesn't currectly work
print_state_vals(linear_env, value_sarsa)
print_policy(linear_env, policy_sarsa)
linear_env.render(policy_sarsa, value_sarsa)
