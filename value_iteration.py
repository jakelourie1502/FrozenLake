import numpy as np
from frozen_lake import gridWorld
import copy
np.set_printoptions(suppress=True, precision=2)
from utils import small_lake, print_state_vals, print_policy

""""
DESCRIPTION
Tolerance level theta, optimal policy achieved when theta -> 0 
Set all S values to 0 
for _ in range(max_iterations) (or just while True)
for all states
    set v_old to V(s)
    V(s) set to max(Qsa) {iterate over actions and take the max}
    if any of the states have a difference in value of over theta, keep going. 

then set the policy to the a that maximises Qsa for each state.
"""

def Qsa(state_values, stateIdx, action, gamma):
    """
    Overview: Function returns the estimated immediate reward (r) and discounted future reward(gamma*V(S_t+1))
    Input: Policy dict with state values, current stateIdx, action chosen, discout rate(gamma)
    
    Method: 
      for each potential next state, calculates chance of transitioning to that state
      computes reward of moving to that state, multiplied by P(S' | S,A) and adds to running reward total
      computes future value of that state, multiplied by P(S' | S,A) and discount rate and adds to running future reward total
    
    Outputs: summed values of reward and future reward
    """
    running_total_val_r =0
    running_total_val_future_r = 0 
    for ns in range(env.n_states):
        ns_probs = env.p(ns, stateIdx, action)
        running_total_val_r += ns_probs * env.r(ns, stateIdx)
        running_total_val_future_r += ns_probs * gamma * state_values[ns]
    q_val = running_total_val_r + running_total_val_future_r
    return q_val

def value_iteration(env,gamma, theta):
    state_values = np.zeros(env.n_states,dtype=float)
    deviation = 100
    counter = 0
    
    while deviation > theta:
        
        counter +=1
        deviation = 0
        ### This section iterates over each state, and calculates the Q value of each action and sets V(s) to max(Qsa)
        for s in range(env.n_states):
            state_values_copy = copy.deepcopy(state_values) #so we're using t-1 state_values
            old_v = state_values_copy[s]
            best_q_val = -float('inf')
            for a in range(env.n_actions):
                qval = Qsa(state_values_copy, s, a, gamma)
                if qval > best_q_val:
                    best_q_val = qval
            state_values[s] = best_q_val
            deviation = max(deviation, abs(best_q_val - old_v))
    
    ### This section then sets the policy based on max Qsa for each state
    policy_max_a = np.zeros(env.n_states,dtype=int)    
    for s in range(env.n_states):
        best_action = 0
        for a in range(env.n_actions):
            best_q_val = -float('inf')
            for a in range(env.n_actions):
                qval = Qsa(state_values_copy, s, a, gamma)
                if qval > best_q_val:
                    best_q_val = qval
                    best_action = a
        policy_max_a[s] = best_action

    return state_values, policy_max_a, counter

if __name__ == '__main__':
    theta = 0.01
    gamma = 0.9
    size, lakes, goals, dist = small_lake()
    env=gridWorld(size,lakes,goals, n_actions = 4, max_steps = 100, dist = dist, seed = None, rnd=0.1)
    state_values, policy_max, counter = value_iteration(env, gamma, theta)
    print_state_vals(env, state_values)
    print_policy(env, policy_max)