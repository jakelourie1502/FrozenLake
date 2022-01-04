import numpy as np
from IPython.display import clear_output

class EnvironmentModel:
    
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]

        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state)

        return next_state, reward

class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, dist, seed = None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps
        
        self.dist = dist
        if self.dist is None:
            self.dist = np.full(n_states, 1./n_states) #returns even distribution
            
    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p = self.dist)
        
        return self.state
    
    
    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid_action.')
            
        self.n_steps += 1
        done = (self.n_steps >= self.max_steps) or self.state == self.terminal_state
        
        self.state, reward = self.draw(self.state, action)
        
        return self.state, reward, done

class gridWorld(Environment):
    
    def __init__(self, size,lakes,goals, n_actions = 4, max_steps = 100, dist = None, seed = None, rnd=0.1):
        n_states = (size[0]*size[1])+1
        Environment.__init__(self, n_states, n_actions, max_steps,dist)
        self.create_dicts_and_indexes(size,lakes,goals)
        self.create_board(size, lakes,goals)
        self.action_dict = {"Up":0, "Right":1, "Down":2,"Left":3}
        self.chance = rnd
        self._init_probs_dict()
        self.reset()
        self.rnd = rnd
        
        
    def p(self,next_state, state, action):
        """
        Here, based on a 'chosen' action, we give the probability of transitioning from one state to another
        Functions:
          We calculate the probability if the chosen action is the 'actual' action, multiplied by chance of not taking random action
          We then add the probability for each action, multiplied by chance of taking random action / number of actions
        """
        no_rnd = 1 - self.rnd
        probas = 0
        probas += no_rnd * self.SAS_probs[state][action][next_state]
        for a in range(self.n_actions):
            probas += (self.rnd/self.n_actions) * self.SAS_probs[state][a][next_state]
        return probas
        "The method p returns the probability of transitioning from state to next state given action. "
        
    def r(self, next_state, state):
        "The method r returns the expected reward in having transitioned from state to next state given action."
        return self.goal_states_idx[state] if state in self.goal_states_idx else 0
    
    def game_play_render(self):
        board = self.board.copy()
        posR, posC = self.stateIdx_to_coors[self.state]
        board[posR, posC] = "P"
        clear_output()
        print(board)
    
    def render(self, policy, value):
        self.print_policy(policy)
        self.print_state_vals(value)

    def print_policy(self, policy_array):
        
        """
        Changes [0,1,2,3] to ["^", ">", "<", "v"]
        Prints the policy
        """
        action_dict = {0:"^", 1: ">", 2: "v", 3: "<"}
        policy_board = np.zeros((self.h , self.w), dtype = str)
        for s in range(self.n_states-1):
            coors = self.stateIdx_to_coors[s]
            policy_board[coors] = action_dict[policy_array[s]]
        print(policy_board)
    
    def print_state_vals(self, values):
        policy_board = np.zeros((self.h , self.w))
        for s in range(self.n_states-1):
            coors = self.stateIdx_to_coors[s]
            policy_board[coors] = values[s]
        print(policy_board)

    def create_dicts_and_indexes(self,size, lakes, goal_states, terminal_state = True):
        """
        Inputs... 
         size of lake (tuple e.g. (4,4))
         Location of lakes in coordinate form e.g. [(0,1),(1,2)...]
         Location of goal_states and their rewards e.g. {(3:3):1, (5,5):-1} In our examples this is always just one goal state

        Outputs...
         Dictionary linking coordinates to index of each state, and reverse dictionary
         Lake squares in index form e.g. [3,6,9]
         Goal states in index form e.g {15: 1, 25: -1}
        """
        self.lakes = lakes
        self.goal_states = goal_states
        self.h = size[0]
        self.w = size[1]
        self.coors_to_stateIdx = {}
        idx =0
        for r in range(self.h):
            for c in range(self.w):
                self.coors_to_stateIdx[(r,c)] = idx

                idx+=1

        if terminal_state:
            self.coors_to_stateIdx[(-1,-1)] = self.n_states-1
            self.terminal_state = self.n_states-1

        self.stateIdx_to_coors = {}
        for k,v in self.coors_to_stateIdx.items():
            self.stateIdx_to_coors[v]=k
        self.lakes_idx = [self.coors_to_stateIdx[x] for x in lakes]
        self.goal_states_idx = {self.coors_to_stateIdx[k]:v for k,v in goal_states.items()}


    def create_board(self,size, lakes, goal_states):
        """
        Inputs: size of lake (h and w), coordinate location of lakes, and coordinate location and value of goal states
        Outputs: array of player-less board, with lake locations and reward locations
        """
        ### Creation of board object
        h,w = size[0],size[1]
        self.board = np.array(['_'] * h*w).reshape(h,w)
        for l in lakes:
            self.board[l] = 'L'
        for g, r in goal_states.items():
            self.board[g] = r
    
    def _init_probs_dict(self):
        """
        In: the backend of the board (stateIdx_to_coors dict, lakes, goals, terminal state)
        Out: returns the impact of an ACTUAL action on the board position of a player
        Structure of output: {Current_State1: {Up: state1, state2, state 3....,
                            Down: state1, state2, state 3...}
                            ....
                    Current_State2: {Up ......}}
        
        note: 'actual' action distinguished here from 'chosen' action. Players 'choose', then we apply randomness, and then there is an 'actual' action
        This function concerns the effect of an 'actual' action on the position of a player.
        """
        
        ### HELPER FUNCTIONS
        def state_is_top(state):
            return self.stateIdx_to_coors[state][0] == 0
        def state_is_bottom(state):
            return self.stateIdx_to_coors[state][0] == self.h-1
        def state_is_left(state):
            return self.stateIdx_to_coors[state][1] == 0
        def state_is_right(state):
            return self.stateIdx_to_coors[state][1] == self.w-1
        def move_up():
            return -self.w
        def move_down():
            return self.w
        def move_left():
            return -1
        def move_right():
            return 1
        
        SA_prob_dict = {}
        lakes_and_goals = list(self.goal_states_idx.keys()) + self.lakes_idx
        
        for state in range(self.n_states):
            SA_prob_dict[state] = {}
            #### Set the chance of entering an absorbing from lake or goal to 1
            for i in range(4):
                SA_prob_dict[state][i] = np.zeros((self.n_states,))
                if state in lakes_and_goals or state == self.terminal_state:
                    for act in range(4):
                        SA_prob_dict[state][i][self.terminal_state] = 1
            
            if state not in lakes_and_goals and state != self.terminal_state:
                """For UP"""
                if not state_is_top(state): #if you're in a normal state, you'll just go up 1
                    SA_prob_dict[state][self.action_dict['Up']][state+move_up()] = 1
                else:
                    SA_prob_dict[state][self.action_dict['Up']][state] = 1

                """For DOWN"""
                if not state_is_bottom(state): #if you're in a normal state, you'll just go up 1
                    SA_prob_dict[state][self.action_dict['Down']][state+move_down()] = 1
                else:
                    SA_prob_dict[state][self.action_dict['Down']][state] = 1

                """For LEFT"""
                if not state_is_left(state): #if you're in a normal state, you'll just go up 1
                    SA_prob_dict[state][self.action_dict['Left']][state+move_left()] = 1
                else:
                    SA_prob_dict[state][self.action_dict['Left']][state] = 1

                """For RIGHT"""
                if not state_is_right(state): #if you're in a normal state, you'll just go up 1
                    SA_prob_dict[state][self.action_dict['Right']][state+move_right()] = 1
                else:
                    SA_prob_dict[state][self.action_dict['Right']][state] = 1     
        self.SAS_probs = SA_prob_dict


if __name__ == '__main__':
    size = (4,4)
    lakes = [(1,1),(1,3),(2,3),(3,0)]
    goals = {(3,3):1}
    dist = np.zeros((size[0]*size[1]+1))
    dist[0]=1
    env=gridWorld(size,lakes,goals, n_actions = 4, max_steps = 100, dist = dist, seed = None, rnd=0.1)
    print(env.board)