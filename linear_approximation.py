import numpy as np
import random


class LinearWrapper:
    def __init__(self, env):
        self.env = env
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states  # 68 (s,a) combinations

    def encode_state(self, s):
        # The method encode state is responsible for representing a state by
        # a one hot encoded feature vector representing (s,a)
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0
        return features

    def decode_policy(self, theta):
        # For each state, returns the greedy policy together with its value function estimate.
        # Used upon training of the model.
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)  # (n_actions, n_features)
            q = features.dot(theta)  # (n_actions, )

            policy[s] = np.argmax(q)
            value[s] = np.max(q)
        return policy, value

    def reset(self):
        # The method reset restarts the interaction between the agent and the environment by setting the number
        # of time steps to zero and drawing a state according to the probability distribution over initial states.
        return self.encode_state(self.env.reset())

    def step(self, action):
        # The method step receives an action and returns a next state drawn according to p, the corresponding
        # expected reward, and a done flag variable.
        state, reward, done = self.env.step(action)

        return state, reward, done

        # return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        # The method render is capable of rendering the state of the environment or a pair of policy and value function.
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    '''
    The function receives an environment (wrapped by LinearWrapper), the maximum number of episodes,
    an initial learning rate, a discount factor, an initial exploration factor, and an (optional) seed
    and returns the parameters of the linear model
    '''

    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)  # array of learning rate, decaying linearly to 0 on max_episodes
    epsilon = np.linspace(epsilon, 0, max_episodes)  # array of epsilon, decaying linearly to 0 on max_episodes
    theta = np.zeros(env.n_features)
    states_visited = set()
    rewards_received = 0
    for i in range(max_episodes):
        eta_episode = eta[i]
        epsilon_episode = epsilon[i]
        state = env.env.reset()  # returns a random state from environment
        while state != env.env.terminal_state:
            features = env.encode_state(state)
            q = features.dot(theta)  # (n_actions, )
            max_ = np.max(q)
            # The ε-greedy policy based on Q should break ties randomly between actions that maximize Q
            index = random.choice([i for i in range(len(q)) if q[i] == max_])
            a = np.random.choice([0, 1, 2, 3]) if random_state.rand() < epsilon_episode else index
            next_state, reward, _ = env.step(a)  # state is int, reward is int
            features_prime = env.encode_state(next_state)  # features_prime (n_actions, n_features) ,
            q_prime = features_prime.dot(theta)  # (n_actions, )
            max_prime = np.max(q_prime)
            index = random.choice([i for i in range(len(q_prime)) if q_prime[i] == max_prime])
            a_prime = np.random.choice([0, 1, 2, 3]) if random_state.rand() < epsilon_episode else index
            delta = reward + gamma * q_prime[a_prime] - q[a]
            theta += eta_episode * delta * features[a]
            state = next_state
            states_visited.add(state)
            rewards_received += reward

    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    '''
    The function receives an environment (wrapped by LinearWrapper), the maximum number of episodes,
        an initial learning rate, a discount factor, an initial exploration factor, and an (optional) seed
        and returns the parameters of the linear model
    '''
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)  # array of learning rate, decaying linearly to 0 on max_episodes
    epsilon = np.linspace(epsilon, 0, max_episodes)  # array of epsilon, decaying linearly to 0 on max_episodes

    theta = np.zeros(env.n_features)
    states_visited = set()
    rewards_received = 0
    for i in range(max_episodes):
        eta_episode = eta[i]
        epsilon_episode = epsilon[i]
        state = env.env.reset()  # returns a random state from environment
        while state != env.env.terminal_state:
            features = env.encode_state(state)
            q = features.dot(theta)  # (n_actions, )
            max_ = np.max(q)
            # The ε-greedy policy based on Q should break ties randomly between actions that maximize Q
            index = random.choice([i for i in range(len(q)) if q[i] == max_])
            a = np.random.choice([0, 1, 2, 3]) if random_state.rand() < epsilon_episode else index
            next_state, reward, _ = env.step(a)  # state is int, reward is int
            features_prime = env.encode_state(next_state)  # features_prime (n_actions, n_features) ,
            q_prime = features_prime.dot(theta)  # (n_actions, )
            a_prime = np.argmax(q_prime)
            delta = reward + gamma * q_prime[a_prime] - q[a]
            theta += eta_episode * delta * features[a]
            state = next_state
            states_visited.add(state)
            rewards_received += reward

    return theta
