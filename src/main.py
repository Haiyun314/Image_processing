import os
from gradient import *
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Get the root project directory
    root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    # Construct the path to the image
    image_path = os.path.join(root_directory, 'images', 'cameraman_sp.png')
    noise_image = plt.imread(image_path)
    
    diff_init = Diff()
    grad_im = diff_init.grad(noise_image)
    lap_im = diff_init.lapl(noise_image)


    grad_norm = np.sum(grad_im[0] ** 2 + grad_im[1] ** 2)
    laplace_product = -np.sum(noise_image * lap_im)


    solution_grad = tykhonov_gradient(noise_image, 0.01, 1000, diff_init)
    solution_four = tykhonov_fourier_denoise(noise_image,0.5)

    _, ax = plt.subplots(1, 3, figsize = (9, 3))
    ax[0].imshow(noise_image, cmap= 'gray')
    ax[0].set_title('noise_image')
    ax[1].imshow(solution_grad, cmap= 'gray')
    ax[1].set_title('tykhonov_gradient')
    ax[2].imshow(solution_four, cmap= 'gray')
    ax[2].set_title('tykhonov_fourier_denoise')
    plt.savefig(os.path.join(root_directory, "results", "denoising.png"))
    plt.show()
    