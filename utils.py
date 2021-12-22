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

def small_lake():
    size = (4,4)
    lakes = [(1,1),(1,3),(2,3),(3,0)]
    goals = {(3,3):1}
    dist = np.zeros((size[0]*size[1]+1))
    dist[0]=1
    return size, lakes, goals, dist