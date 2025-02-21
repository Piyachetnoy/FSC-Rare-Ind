import numpy as np
import matplotlib.pyplot as plt

# Load the density map from the .npy file
density = np.load('dataTest/gt_density_map_adaptive_384_VarV2/154.npy')

# Visualize the density map
plt.imshow(density, origin='lower', extent=[-3, 3, -3, 3])
plt.colorbar()
plt.show()