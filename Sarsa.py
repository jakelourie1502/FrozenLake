import numpy as np



def sarsa(env,max_episodes,eta,gamma,epsilon,optimal_policy=None,seed=None, initial_q=0,eta_floor = 0,
          epsilon_floor=0,epsilon_ramp_epoch=None,eta_ramp_epoch=None,madness=1):
    # Get random state
    #random_state = np.random.RandomState(seed)

    # Initialise learning rate and probability of random action
    # eta and epsilon decrease linearly as number of episodes increases
    if eta_ramp_epoch==None:
        eta_ramp_epoch=max_episodes

    eta = np.linspace(eta, eta_floor, eta_ramp_epoch)

    if epsilon_ramp_epoch==None:
        epsilon_ramp_epoch=max_episodes

    epsilon = np.linspace(epsilon, epsilon_floor, epsilon_ramp_epoch)
    
    # Set Q values for each state and action to zero
    q = np.zeros((env.n_states,env.n_actions))+initial_q
    episodes = 0
    for i in range(max_episodes):
        #print(i)
        episodes += 1
        need_for_madness = np.random.randint(0,madness,1)
        # Initial state for episode i
        state = env.reset() # dist is none - chooses random state
        
        ######## Write Code Here ############
        #set minimum eta and epsilon
        if episodes>epsilon_ramp_epoch:
            epz = epsilon_floor
        else:
            epz = epsilon[i]

        if episodes>eta_ramp_epoch:
            eeta = eta_floor
        else:
            eeta = eta[i]

        # Select action a for state s accoridng to an e-greedy policy based on Q
        if epz > np.random.rand(1)[0]:
            action = np.random.choice(np.array([0,1,2,3]))
        else:
            action = np.argmax(q[state,:])

        done = False
        step = 0
        while not done: # Check State is not terminal
            step +=1
            # Get next_state, reward and done flag for action a at state s
            next_obs_state, reward, done = env.step(action)
            #print(next_obs_state,reward,done)
            # Select action a' for state s' according to an e-greedy policy based on Q
            if epz > np.random.rand(1)[0] or step<need_for_madness:
                next_action = np.random.choice(np.array([0,1,2,3]))
            else:
                next_action = np.argmax(q[next_obs_state, :])

            if step>need_for_madness:
                # Update q table
                q[state,action]=q[state,action]+eeta*(reward+gamma*(q[next_obs_state,next_action])-q[state,action])


            # Set next state and next action
            state = next_obs_state
            action = next_action


        # Update policy and value
        policy = q.argmax(axis=1)
        value = q.max(axis=1)

        if type(optimal_policy) == np.ndarray:
            check_opt,wrong = check_optimal(optimal_policy, policy, env)
            if episodes % 2500 == 0:
                print('Number of Episodes:', episodes)
                print('Incorrect Policies:', wrong)
                env.render(policy, value)
            if check_opt:
                break

    return policy, value, episodes



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