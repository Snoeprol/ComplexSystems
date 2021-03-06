import numpy as np
import numba as nb
from scipy.stats import maxwell
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import copy

# Periodic boundary conditions
# Rabbits are field with positions, velocities awareness radius.
NVAR = 5
N = 50
R = 0.15
GA = 0.2
RA = 1
PA = 1
dt = 0.01
ms = 5
RADIUS_PRED = 0.2
Niterations = 1000
MAX_SPEED = 0.01/dt
DECAY_RATE = 0.995

#@nb.njit
def init_rabbits(N, T):
    # Velocities and positions
    rabbits = np.zeros((N, NVAR))

    positions = np.random.rand(N, 2)
    rabbits[:, 0:2] = positions
    
    velocities = 0#T * (np.random.rand(N, 2) - 1/2)
    rabbits[:, 2:4] = velocities
    
    return rabbits

def init_nn_array(N):
    return np.zeros((N, N - 1))

@nb.njit
def get_rn_noise(amplitude):
    
    # Random walk
    N = len(rabbits)
    perturbation = amplitude * (np.random.rand(N, 2) - 0.5)
    return perturbation

@nb.njit
def get_predator_force(N, rabbits, predator, PA):
    distances = rabbits[:, 0:2] - predator[0:2]
    # Inefficient
    distances = np.where(distances > RADIUS_PRED, distances, np.inf)
        
    return PA/distances

def update_velocities(dt):
    nn = get_nn(R)
    group_force = get_group_influence(GA, nn)
    perturbation = get_rn_noise(RA)
    predator_force = get_predator_force(N, rabbits, predator, PA)
    
    rabbits[:, 2:4] += dt * (group_force + perturbation + predator_force) 
    rabbits[:, 2:4] *= DECAY_RATE

    for rabbit in rabbits:
        v = np.linalg.norm(rabbit[2:4])
        if v > MAX_SPEED:
            rabbit[2:4] = rabbit[2:4] * MAX_SPEED/ v

def step(dt):
    
    update_velocities(dt)
    rabbits[:, 0] = (rabbits[:, 0] + dt * rabbits[:, 2]) % 1
    rabbits[:, 1] = (rabbits[:, 1] + dt * rabbits[:, 3]) % 1
    predator[0] = (dt * predator[2]) % 1
    predator[1] = (dt * predator[3]) % 1
    return rabbits

def get_nn(r):
    """
    Returns array nn_indices, where the N-th elements contains the indices of the N-th rabbit
    
    shape nn_indices = N
    """
    
    # Clean arrays
    nn_array[:, :] = -1
    combination_array[:, :] = 0
    nn_indices[:] = 0
    
    # For loop has O(N^@)
    for i, rabbit in enumerate(rabbits):
        for j, rabbit_n in enumerate(rabbits):
            # Remove self-interaction
            if i != j:
                # Remove redundant calculations
                if combination_array[i, j] != 1:
                    combination_array[i, j] = 1
                    combination_array[j , i] = 1
                    # Possible new swarm interaction
                    d = np.linalg.norm(rabbit[0:2] - rabbit_n[0:2])
                    if d < r:
                        nn_array[i, nn_indices[i], 0] = j
                        nn_array[j, nn_indices[j], 0] = i
                        nn_array[i, nn_indices[i], 1] = d
                        nn_array[j, nn_indices[j], 1] = d
                        nn_indices[i] += 1
                        nn_indices[j] += 1
                        
    return nn_array


def animatePeriodic(i):
    step(dt)
    rect.set_edgecolor('k')
    field.set_data(rabbits[:, 0],rabbits[:, 1])
    return field,rect

#@nb.njit
def get_group_influence(amplitude, nn):
    """Given the nearest neighbor array, calculates the swarm influence 

    Parameters
    ----------
    rabits : float array
        Array containing all values of positions and velocities of the rabbits.
    nn_array : int, float array
        Array where N-th elements contains the indices combined with distances to nearest neighbors.

    Returns
    -------
    float array
        N, 2 float array which describes the force caused by the group
    """
    
    # TODO Add weighted sums (closer neighbors have more influence)
    #total_weights = np.zeros((N))
    
    
    group_velocities = np.zeros((N, 2))
    
    # Collect weights
    #for i, rabbit in enumerate(nn):
    #    total_weights[i] = np.sum(rabbit[:, 0])
    
    temp_vel = np.zeros((2))
    # Calculate the average velocity vector of neighbors weighed proportionally to their distance
    for i, rabbit in enumerate(nn):
        # No neighbors
        #if total_weights[i] == 0:
            #break
        
        temp_vel[:] = 0
        for neighbor in rabbit:
            if neighbor[0] == -1:
                break
            else:
                # Neighbor 0 is index
                # Neighbor 1 is distance
                temp_vel += rabbits[neighbor[0], 2:4]
        #temp_vel /= total_weights[i]
        # TODO MIGHT NEED TO ADD COPY AGAIN
        group_velocities[i] = temp_vel
        temp_vel[:] = 0
        
    return amplitude * group_velocities          
            
        
if __name__ == '__main__':
    rabbits = init_rabbits(N, 2)
    predator = np.array([0.5, 0.5, 0.1, 0.1])
    nn_array = np.zeros((N, N, 2), dtype = np.int)
    combination_array = np.zeros((N, N), dtype = np.int)
    nn_indices = np.zeros((N), dtype = np.int)

    
    fig = plt.figure()
    
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                        xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))

    # field holds the locations of the field
    field, = ax.plot([], [], 'bo', ms=6)
    field.set_markersize(ms)

    rect = plt.Rectangle([0,0],1,1,
                        ec='none', lw=2, fc='none')
    ax.add_patch(rect)
    
    ani = animation.FuncAnimation(fig, animatePeriodic, frames=Niterations,
                                interval=1, blit=True)
    plt.show()
