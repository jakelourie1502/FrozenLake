import numpy as np

def check_optimal(optimal_policy,policy,env):
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
    optimal_policy = np.array(policy)
    lakes = env.lakes_idx
    goals = np.array(list(env.goal_states_idx.keys()))
    terminals = goals + 1
    exclude = np.append(lakes, np.append(goals, terminals))
    optimal_policy = np.delete(optimal_policy, exclude)
    return optimal_policy
