import numpy as np



def sarsa(env,max_episodes,eta,gamma,epsilon,seed=None):
    # Get random state
    random_state = np.random.RandomState(seed)

    # Initialise learning rate and probability of random action
    # eta and epsilon decrease linearly as number of episodes increases
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    
    # Set Q values for each state and action to zero
    q = np.zeros((env.n_states,env.n_actions))
    episodes = 0
    for i in range(max_episodes):
        #print(i)
        episodes += 1
        # Initial state for episode i
        state = env.reset()
        
        ######## Write Code Here ############
        # Select action a for state s accoridng to an e-greedy policy based on Q
        if epsilon[i] > np.random.rand(1)[0]:
            action = np.random.choice(np.array([0,1,2,3]))
        else:
            action = np.argmax(q[state,:])

        done = False

        while not done: # Check State is not terminal
            # Get next_state, reward and done flag for action a at state s
            next_obs_state, reward, done = env.step(action)
            #print(next_obs_state,reward,done)
            # Select action a' for state s' according to an e-greedy policy based on Q
            if epsilon[i] > np.random.rand(1)[0]:
                next_action = np.random.choice(np.array([0,1,2,3]))
            else:
                next_action = np.argmax(q[next_obs_state, :])

            # Update q table
            q[state,action]=q[state,action]+eta[i]*(reward+gamma*(q[next_obs_state,next_action])-q[state,action])


            # Set next state and next action
            state = next_obs_state
            action = next_action


    # Update policy and value
    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value, episodes
