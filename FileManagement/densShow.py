import numpy as np
import matplotlib.pyplot as plt

# Load the density map from the .npy file
density = np.load('./data-final/density_map_adaptive_V1/332.npy')

# Visualize the density map
plt.imshow(density, origin='lower', extent=[-3, 3, -3, 3])
plt.colorbar()
plt.show()