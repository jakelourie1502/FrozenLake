import numpy as np
from frozen_lake import gridWorld
import copy
from Default_Lakes import small_lake, big_lake
import time
"""
Policy evaluation should be in the form a dictionary...
{
    state1:{
        policy: [(p(a))],  ##probabability of each action given state
        action_values: [(G(a))],  ##value of each action given state
        Value: V(S_π) ##value of policy for that state
    }
    ,
    state2: {
        policy: [(p(a))], 
        action_values: [(G(a))], 
        Value: V(S_π)
    }
}
"""

def Qsa(env,Policy_info, stateIdx, action, gamma):
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


def one_time_evaluation(Policy_info,env,gamma):
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
            reward, future_val = Qsa(env,Policy_info, stateIdx, action, gamma)
            Policy_info_copy[stateIdx]['action_value'][action] = reward + future_val
            running_total_val += (reward + future_val) * Policy_info_copy[stateIdx]['policy'][action]
       
        Policy_info_copy[stateIdx]['value'] = running_total_val
    return Policy_info_copy, old_policy
    

def policy_evaluation_loop(starting_policy_info, env, method = 'converge', iterations=100, theta=0.1,gamma=0.9):
    cur_policy = starting_policy_info.copy()
    it = 0
    
    if method == 'iteration':
        for i in range(iterations):
            cur_policy, _ = one_time_evaluation(cur_policy, env)
        it = iterations
    if method == 'converge':
        
        while it < iterations:
            it+=1
            new_policy, old_policy = one_time_evaluation(cur_policy, env,gamma)
            loss_total =0
            for old, new in zip(old_policy.values(), new_policy.values()):
                l = abs(old['value'] - new['value'])
                loss_total+=l
            
            cur_policy = new_policy.copy()
            
            if loss_total < theta:
                break
    return cur_policy, it

def policy_improvement(env, policy_info):
    """This function improves on a previous policy by selecting the best action in a given state given policy evaluation"""
    policy = np.zeros((env.n_states, env.n_actions), dtype=int)
    idx = 0
    for s, state in policy_info.items():
        policy[idx][np.argmax(state['action_value'])] = 1
        idx += 1
    return policy

def policy_iteration(env, gamma, theta, max_it_iter, max_it_eval = 50, eval_method = 'converge', policy_info = None):
    """
    max_it_eval: number of iterations each time a policy evaluation is carried out
    max_it_iter: number of iterations of the overall algorithm
    """
    if policy_info == None:
        policy_info = init_blank_policy(env)

    for it in range(max_it_iter):
        policy_info, _ = policy_evaluation_loop(policy_info, env, method = eval_method, iterations=max_it_eval, theta=theta, gamma = gamma)
        pol = policy_improvement(env, policy_info)
        no_change_checker = True
        for s in range(env.n_states):                
            if not np.array_equal(pol[s], policy_info[s]['policy']):
                no_change_checker = False    
                policy_info[s]['policy'] = pol[s]

        if no_change_checker:
            print(f"Policy static after {it+1} iterations")
            state_values = [policy_info[x]['value'] for x in range(env.n_states)]
            policy_max = [np.argmax(policy_info[x]['policy']) for x in range(env.n_states)]
            return policy_max, state_values, it+1
    print("Policy didn't converge")
    state_values = [policy_info[x]['value'] for x in range(env.n_states)]
    policy_max = [np.argmax(policy_info[x]['policy']) for x in range(env.n_states)]
    return policy_max, state_values, max_it_iter

def init_blank_policy(env):
    Policy_info = {}
    for s in range(env.n_states):
        Policy_info[s] = {}
        Policy_info[s]['policy'] = np.zeros((env.n_actions,)) + 1 / env.n_actions
        Policy_info[s]['action_value'] = np.zeros((env.n_actions,))
        Policy_info[s]['value'] = 0
    return Policy_info

if __name__ == '__main__':
    
    np.set_printoptions(suppress=True, precision=2)
    ## Load in the environment
    size, lakes, goals, dist = small_lake()
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
    max_overall_its = 10
    theta = 0.001
    tn = time.time()
    policy, state_values, counter = policy_iteration(env, gamma, theta, max_overall_its)
    tn = time.time() - tn
    print(f'It took {counter} iterations and {tn} seconds')
    
    env.render(policy,state_values)
    