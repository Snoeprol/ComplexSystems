import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import bisect 

def get_normalized_size(size, factor):
    return factor * size

# different gammas and betas list, with classification
gammatjes = gamma = np.linspace(0,1,11)
beta0 = gamma/(1-gamma)
beta1 = (1+gamma)/(1-gamma)
beta2 = (beta1 + beta0)/2
values0 = np.array(["eq","ex","ex","ex","ex","ex","ex","ex","eq","eq","nan"])
values1 = np.array(["ex","osc","osc","osc","osc","osc","osc","osc","osc","osc","nan"])
values2 = np.array(["ex","eq","eq","eq","eq","eq","eq","eq","eq","eq","nan"])
colors = ["tab:blue","tab:orange", "tab:red"]
len(values0), len(values1),len(values2),np.linspace(0,1,11)

betatjes = np.array([beta0,beta1,beta2])
beta = betatjes
values = np.array([values0,values1,values2])
types = ["eq","ex","nan","osc"]
markers = ["_","x","d","^"]

gamma_plot = np.linspace(0,1,51)

line0 = plt.plot(gamma_plot, gamma_plot/(1-gamma_plot), label=r"$\beta_0$", color="tab:blue")
line1 = plt.plot(gamma_plot, (1+gamma_plot)/(1-gamma_plot), label=r"$\beta_1$", color="tab:orange")
line2 = plt.plot(gamma_plot, (1+2*gamma_plot)/(2*(1-gamma_plot)), label=r"$\frac{\beta_0+\beta_1}{2}$", color="tab:red")

# Correction factor of circle with radius 60
factor = 500**2 / (500**2 - np.pi * 60**2)

gammas = np.zeros((11))
gammas[1:] = np.linspace(0.1, 1, 10)
gammas[0] = 0.05
betas = np.round(np.linspace(0.1, 15.1, 15), 1)

x_ticks = gammas
y_ticks = betas
fig, axes = plt.subplots(1, 2, figsize = (15,5))

title_string = ['With circle', 'Without circle']
csvs = ['results_yes_circle.csv', 'results_no_circle.csv']
sizes = []

# Initial loop to extract all the sizes
for i, csv_i in enumerate(csvs):
    data = pd.read_csv(csv_i)


    beta_indices = []
    gamma_indices = []
    for index, row in data.iterrows():
        beta = float(row['beta']) 
        if beta not in beta_indices:
            bisect.insort(beta_indices, beta)
        gamma = float(row['gamma'])
        if gamma not in gamma_indices:
            bisect.insort(gamma_indices, gamma)
        size = row['size'] 
        if i == 0:
            size = get_normalized_size(size, factor)
        sizes.append(size)


# Do plots
for i, csv_i in enumerate(csvs):
    data = pd.read_csv(csv_i)


    beta_indices = []
    gamma_indices = []
    for index, row in data.iterrows():
        beta = float(row['beta']) 
        if beta not in beta_indices:
            bisect.insort(beta_indices, beta)
        gamma = float(row['gamma'])
        if gamma not in gamma_indices:
            bisect.insort(gamma_indices, gamma)
        size = row['size']
        if i == 0:
            size = get_normalized_size(size, factor)
    plotting_array = np.zeros((len(beta_indices), len(gamma_indices)))

    for index, row in data.iterrows():
        beta = float(row['beta']) 
        gamma = float(row['gamma'])
        gamma_index = gamma_indices.index(gamma)
        beta_index = beta_indices.index(beta)
        size = row['size']
        # Only correct when i = 0 == with circle.
        if i == 0:
            size = get_normalized_size(size, factor)
        plotting_array[beta_index, gamma_index] = float(size) - min(sizes)
        
    im1 = axes[i].imshow(plotting_array, label = str(i), vmin = 0, vmax = 1e6, origin ='bottom', aspect = 'auto', extent = [0, 1, 0, 15])
    fig.colorbar(im1, ax=axes[i])
    axes[i].set_xticks(x_ticks)
    axes[i].set_yticks(y_ticks)
    axes[i].set_title(title_string[i])
    line0 = axes[i].plot(gamma_plot, gamma_plot/(1-gamma_plot), label=r"$\beta_0$", color="tab:blue")
    line1 = axes[i].plot(gamma_plot, (1+gamma_plot)/(1-gamma_plot), label=r"$\beta_1$", color="tab:orange")
    axes[i].set_ylim([0, 15])
    axes[i].legend()
plt.show()


