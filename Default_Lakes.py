import numpy as np
def small_lake():
    size = (4,4)
    lakes = [(1,1),(1,3),(2,3),(3,0)]
    goals = {(3,3):1}
    dist = np.zeros((size[0]*size[1]+1))
    dist[0]=1
    return size, lakes, goals, dist

def big_lake():
