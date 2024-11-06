import numpy as np
import matplotlib.pyplot as plt
import os
from gradient import Diff

def chan_vese(img, max_iter=300, tol=1e-3, mu=0.2, dt=0.1):
    """
    Chan-Vese segmentation algorithm
    Parameters:
    - img : 2D array, grayscale image
    - max_iter : maximum number of iterations
    - tol : tolerance for stopping criterion
    - mu : parameter for controlling length of level set curve
    - dt : time step for update

    Returns:
    - phi : final level set function (segmentation boundary)
    """

    # Initialize level set function phi (signed distance function)
    phi = np.ones_like(img)
    phi[img < 0.5] = -1  # Initial contour around brighter objects
    lap = Diff()


    for i in range(max_iter):
        # Compute averages inside and outside the contour
        inside = img[phi > 0]
        outside = img[phi <= 0]
        c1 = inside.mean() if inside.size > 0 else 0
        c2 = outside.mean() if outside.size > 0 else 0

        # Update the level set function
        dphi_dt = - (img - c1)**2 + (img - c2)**2  # Data fitting term
        dphi_dt += mu * lap.lapl(phi)  # Regularization term (smoothness)

        # Evolve the level set
        phi = phi + dt * dphi_dt

        # Reinitialize phi to maintain numerical stability
        phi = np.sign(phi) * np.maximum(np.abs(phi), tol)

        # Convergence check
        if np.linalg.norm(dphi_dt) < tol:
            print(f"Converged at iteration {i}")
            break

    return phi


root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir))
image_path = os.path.join(root_path, 'images', 'cameraman_nse.png')
# Load and preprocess image
img = plt.imread(image_path)
# print(img)
# Apply Chan-Vese
phi = chan_vese(img)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Segmentation Result")
plt.imshow(img, cmap='gray')
plt.contour(phi, levels=[0], colors='r')  # Overlay contour on the image
plt.axis('off')
plt.show()

