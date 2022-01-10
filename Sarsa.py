import numpy as np
from utils import check_optimal



def sarsa(env,max_episodes,eta,gamma,epsilon,optimal_policy=None,seed=None, initial_q=0,eta_floor = 0,
          epsilon_floor=0,epsilon_ramp_epoch=None,eta_ramp_epoch=None,madness=1):
    # Get random state
    random_state = np.random.RandomState(seed)
    """
    eta_floor: after eta_ramp_epochs, set eta to a constant level
    epsilon_floor: after epsilon_ramp_epoch, set epsilon to a constant level
    madness: play random moves for the first X moves in order to create different effective starting points. don't optimise sarsa during these moves
    """
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
        if epz > random_state.rand():
            action = np.random.choice(np.array([0,1,2,3]))
        else:
            max_action = np.max(q[state,:])
            action = np.random.choice(
                [i for i in range(len(q[state, :])) if q[state, i] == max_action])

        done = False
        step = 0
        while not done: # Check State is not terminal
            step +=1
            # Get next_state, reward and done flag for action a at state s
            next_obs_state, reward, done = env.step(action)
            #print(next_obs_state,reward,done)
            # Select action a' for state s' according to an e-greedy policy based on Q
            if epz > random_state.rand() or step<need_for_madness:
                next_action = np.random.choice(np.array([0,1,2,3]))
            else:
                max_next_action = np.max(q[next_obs_state, :])
                next_action = np.random.choice([i for i in range(len(q[next_obs_state, :])) if q[next_obs_state, i] == max_next_action])

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
            if episodes % 5000 == 0:
                print('Number of Episodes:', episodes)
                print('Incorrect Policies:', wrong)
                env.render(policy, value)
            if check_opt:
                break

    return policy, value, episodes