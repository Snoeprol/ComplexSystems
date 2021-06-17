from numba import njit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from tqdm import tqdm

@njit
def init_arrays(N):
    U = np.ones((N, N))#np.random.rand(N, N)
    V = np.zeros((N, N))
    #V[N//2-10 : N//2+10, N//2-10 : N//2+10] = 5
    V[N//5:N//5 + N//10, N//2:N//2 + N//10] = 5
    V[N//2:N//2 + N//10, N//5:N//5 + N//10] = 5
    F = np.zeros((N, N))
    G = np.zeros((N, N))
    L_U = np.zeros((N, N)) 
    L_V = np.zeros((N, N)) 
    return U, V, F, G, L_U, L_V

@njit()
def get_F(U, V, alpha, beta, gamma):
    return alpha * U * ((1 - U) - (V / (1 + beta * U)))

@njit
def get_G(U, V, alpha, beta, gamma):
    return V * (beta * U / (1 + beta * U) - gamma)

@njit
def get_laplacian(D_U, U, L_U):
    """
    L_U is the array that will be filled with the Laplacian
    """
    for i in range(0, N):
        for j in range(0, N):
            L_U[i, j] = D_U * (U[(i + 1) % N, j]+U[(i - 1 + N) % N, j]+U[i, (j + 1) % N]+ U[i, (j - 1 + N) % N]- 4*U[i, j]) 

@njit
def update(A, F_A, L_A, dt):
    """
    Updates a function A where F_A is the updating function of A
    and L_A is the Laplacian of A
    """
    A += dt * (F_A + L_A)
    return A

@njit
def do_iter_diffusion(U, V, F, G, L_U, L_V, D_U, D_V, alpha, beta, gamma, dt, data_array = None, i = None):
    # Initialize concentration matrix c(x,y;t)
    F = get_F(U, V, alpha, beta, gamma)
    G = get_G(U, V, alpha, beta, gamma)
    get_laplacian(D_U, U, L_U)
    get_laplacian(D_V, V, L_V)
    U = update(U, F, L_U, dt)
    V = update(V, G, L_V, dt)
    data_array[i, 0] = np.mean(U)
    data_array[i, 1] = np.mean(V)
    

def animate_func(i):
    tic = time.perf_counter()
    do_iter_diffusion(U, V, F, G, L_U, L_V, D_U, D_V, alpha, beta, gamma, dt, data_array, i)
    toc = time.perf_counter()
    print(f"Updated in {toc - tic:0.4f} seconds")
    im_1 = axarr[0].imshow(U)
    im_2 = axarr[1].imshow(V)
    for i, cbar in enumerate(cbars):
        cbar.set_clim(vmin=0,vmax=1)
        cbar_ticks = np.linspace(0., 2., num=11, endpoint=True)
        cbar.set_ticks(cbar_ticks) 
        cbar.draw_all()
    return [axarr[0], axarr[1]]

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
        
N = 40
U, V, F, G, L_U, L_V = init_arrays(N)
D_U = 0.01
D_V = 1
alpha = 0.5
beta = 12
gamma = 0.5
dt = 0.05

Nit = 10_000
data_array = np.zeros((Nit, 2))

fps = 15
nSeconds = 10

fig, axarr = plt.subplots(1,2)

im_1 = axarr[0].imshow(U, interpolation='none')
im_2 = axarr[1].imshow(V, interpolation='none')

ims = [im_1, im_2]
cbars = []
for im in ims:
    # Create colorbar
    cbar = plt.colorbar(im)
    cbars.append(cbar)


anim = animation.FuncAnimation(fig, animate_func,frames = nSeconds * fps, interval = 1000 / fps, repeat=False)
plt.show()

#anim.save('test_anim.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])

print('Done!')

for i in tqdm(range(Nit)):
    do_iter_diffusion(U, V, F, G, L_U, L_V, D_U, D_V, alpha, beta, gamma, dt, data_array, i)
fig, axes = plt.subplots(1, 3, figsize = (15,5))

axes[0].plot(data_array[:, 0], label = 'U')
axes[0].plot(data_array[:, 1], label = 'V')
axes[0].legend()
im1 = axes[1].imshow(U)
fig.colorbar(im1, ax = axes[1])
axes[1].title.set_text('U')
im2 = axes[2].imshow(V)
fig.colorbar(im2, ax = axes[2])
axes[2].title.set_text('V')

# Create new directory
output_dir = "data"
mkdir_p(output_dir)

plt.savefig(f'{output_dir}/N_{N}_Nit_{Nit}_a_{alpha}_b_{beta}_c_{gamma}_DU_{D_U}_DV_{D_V}_dt_{dt}.png', dpi = 500)
plt.show()

