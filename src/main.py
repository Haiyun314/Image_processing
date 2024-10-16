import os
from gradient import *
import numpy as np
import matplotlib.pyplot as plt
import show_image as si

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

    images = np.stack((noise_image, solution_grad, solution_four), axis= 0)
    names = ['noise_image', 'tykhonov_gradient', 'tykhonov_fourier_denoise']
    si.show_images(images= images, number_of_images=3, names=names, name='denoising', save=1)

    