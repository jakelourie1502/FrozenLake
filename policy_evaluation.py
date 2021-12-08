import numpy as np
from frozen_lake import gridWorld
import copy

"""
Policy evaluation should be in the form a dictionary...
{
    {state1:
        policy: [(p(a))],  ##probabability of each action given state
        action_values: [(G(a))],  ##value of each action given state
        Value: V(S_π) ##value of policy for that state
    }
    ,
    {state2:
        policy: [(p(a))], 
        action_values: [(G(a))], 
        Value: V(S_π)
    }
}
"""

def Qsa(Policy_info, stateIdx, action, gamma):
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
        running_total_val_future_r += ns_probs * gamma * Policy_info[ns]['value']
        
    return running_total_val_r, running_total_val_future_r


def one_time_evaluation(Policy_info,env):
    """
    Overview: Evaluates the value of the policy for each state, returning the overall policy value of a state, and the individual values of actions at states
    Input: Policy_info dictionary and the environment. 
    
    Method:
      Copies policy so we're always using the previous estimate of state values
      For each state, calculate value of action using Qsa function and store this.
      Sum together action values weighted by policy probabiliies to get total value of policy.
    
    Outputs: Updated policy_info dictionary

    """
    Policy_info_copy = copy.deepcopy(Policy_info) #copy so we're not updating states and then updating other states with their updated values.
    old_policy = copy.deepcopy(Policy_info)
    
    for stateIdx, state_info in Policy_info_copy.items():
        running_total_val = 0
        for action in range(env.n_actions):
            reward, future_val = Qsa(Policy_info, stateIdx, action, gamma)
            Policy_info_copy[stateIdx]['action_value'] = reward + future_val
            running_total_val += (reward + future_val) * Policy_info_copy[stateIdx]['policy'][action]
       
        Policy_info_copy[stateIdx]['value'] = running_total_val
    print_policy(old_policy)
    return Policy_info_copy, old_policy
    
def print_policy(Policy_info):
    policy_board = np.zeros((env.h , env.w))
    for s in range(env.n_states-1):
        coors = env.stateIdx_to_coors[s]
        policy_board[coors] = Policy_info[s]['value']
    print(policy_board)

def policy_evaluation_loop(starting_policy_info, env, view_board=True, method = 'iteration', iterations=100, epsilon=0.1,gamma=0.9):
    cur_policy = starting_policy_info.copy()
    print_policy(cur_policy)
    
    if method == 'iteration':
        for i in range(iterations):
            cur_policy, _ = one_time_evaluation(cur_policy, env)
            print_policy(cur_policy)
        it = iterations
    if method == 'converge':
        it = 0
        while True:
            it+=1
            new_policy, old_policy = one_time_evaluation(cur_policy, env)
            print_policy(old_policy)
            loss_total =0
            for old, new in zip(old_policy.values(), new_policy.values()):
                l = abs(old['value'] - new['value'])
                loss_total+=l
            
            cur_policy = new_policy.copy()
            
            if loss_total < epsilon:
                break
    return cur_policy, it

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=2)
    ## Load in the environment
    size = (4,4)
    lakes = [(1,1),(1,3),(2,3),(3,0)]
    goals = {(3,3):1}
    dist = np.zeros((size[0]*size[1]+1))
    dist[0]=1
    env=gridWorld(size,lakes,goals, n_actions = 4, max_steps = 100, dist = dist, seed = None, rnd=0.1)

    ##### Initiliase the policy info dictionary and random policy
    Policy_info = {}
    for s in range(env.n_states):
        Policy_info[s] = {}
        Policy_info[s]['policy'] = np.zeros((env.n_actions,)) + 1 / env.n_actions
        Policy_info[s]['action_value'] = np.zeros((env.n_actions,))
        Policy_info[s]['value'] = 0

    #Init Parameters
    gamma = 0.9
    print_policy_board = True
    method = 'iterations'
    iterations = 100
    epsilon = 0.001
    
    new_pol, it = policy_evaluation_loop(Policy_info, env, view_board=print_policy_board, method='converge',iterations=iterations,epsilon=epsilon,gamma=gamma) 
    print(it)