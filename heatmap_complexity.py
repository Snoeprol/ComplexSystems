import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import bisect 


fig, axes = plt.subplots(1, 2, figsize = (15,5))

title_string = ['With circle', 'Without circle']
csvs = ['results_yes_circle.csv', 'results_no_circle.csv']
for i, csv_i in enumerate(csvs):
    data = pd.read_csv(csv_i)
    sizes = []

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
        sizes.append(size)
        
    plotting_array = np.zeros((len(beta_indices), len(gamma_indices)))

    for index, row in data.iterrows():
        beta = float(row['beta']) 
        gamma = float(row['gamma'])
        gamma_index = gamma_indices.index(gamma)
        beta_index = beta_indices.index(beta)
        size = row['size']
        plotting_array[beta_index, gamma_index] = float(size) - min(sizes)
        
    im1 = axes[i].imshow(plotting_array, label = str(i), vmin = 0, vmax = 1e6, origin ='bottom', aspect = 'auto')
    fig.colorbar(im1, ax=axes[i])

    axes[i].set_title(title_string[i])
plt.show()