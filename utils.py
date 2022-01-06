import numpy as np
def print_policy(env, policy_array):
    """Takes an environment and an array of size (env.n_states, ) 
    ReShapes array to env.h, env.w
    Changes [0,1,2,3] to ["^", ">", "<", "v"]
    Prints the policy
    """
    action_dict = {0:"^", 1: ">", 2: "v", 3: "<"}
    policy_board = np.zeros((env.h , env.w), dtype = str)
    for s in range(env.n_states-1):
        coors = env.stateIdx_to_coors[s]
        policy_board[coors] = action_dict[policy_array[s]]
    print(policy_board)
    
def print_state_vals(env, values):
    policy_board = np.zeros((env.h , env.w))
    for s in range(env.n_states-1):
        coors = env.stateIdx_to_coors[s]
        policy_board[coors] = values[s]
    print(policy_board)



def check_optimal(optimal_policy,policy,env):
    if len(policy)<20:
        policy = np.array(policy)
        truth_array = np.abs(optimal_policy - policy)
        truth = np.sum(truth_array)
        wrong = np.count_nonzero(truth_array)
    else:
        lakes = env.lakes_idx
        goals = np.array(list(env.goal_states_idx.keys()))
        terminals = goals + 1
        exclude = np.append(lakes, np.append(goals, terminals))
        policy = np.array(policy)
        policy = np.delete(policy,exclude)
        truth_array = np.abs(optimal_policy - policy)
        truth = np.sum(truth_array)
        wrong = np.count_nonzero(truth_array)

    if truth == 0:
        return True,wrong
    else:
        return False,wrong


def prep_optimal(env,policy):
    if len(policy)<20:
        optimal_policy = np.array(policy)
    else:
        optimal_policy = np.array(policy)
        lakes = env.lakes_idx
        goals = np.array(list(env.goal_states_idx.keys()))
        terminals = goals + 1
        exclude = np.append(lakes, np.append(goals, terminals))
        optimal_policy = np.delete(optimal_policy, exclude)
    return optimal_policy
