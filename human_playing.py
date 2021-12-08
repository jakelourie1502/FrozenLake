import numpy as np
from frozen_lake import gridWorld
## Load in the environment
size = (4,4)
lakes = [(1,1),(1,3),(2,3),(3,0)]
goals = {(3,3):1}
dist = np.zeros((size[0]*size[1]+1))
dist[0]=1
env=gridWorld(size,lakes,goals, n_actions = 4, max_steps = 100, dist = dist, seed = None, rnd=0.1)
state = env.reset()
env.render()
actions = [0,1,2,3]
done = False

while not done:
    c = int(input('\nMove: '))
    if c not in actions:
        raise Exception('Invalid_action')
        
    state, r, done = env.step(actions.index(c))
    if state == env.terminal_state: 
        done == True; break
        
    env.render()
print(f"Game Over, Reward: {r}")