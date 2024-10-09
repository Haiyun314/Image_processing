import numpy as np
import matplotlib.pyplot as plt

def test_imageinfo(image:np.ndarray) -> None:
    print(image.shape)
    print(np.max(image), np.min(image))

class Diff:
    '''
    boundary condition: periodic
    discretize: central differences for second derivatives (Laplace) and backward differences for gradient
    '''
    def grad(self, image: np.ndarray) -> np.ndarray:
        # Backward differences for gradient (periodic boundary)
        image_hori_shift = np.concatenate((image[..., -1:], image[..., :-1]), axis=-1)
        image_vert_shift = np.concatenate((image[1:, ...], image[:1, ...]), axis=0)

        grad_hori = image - image_hori_shift
        grad_vert = image - image_vert_shift
        return np.stack((grad_hori, grad_vert), axis=0)
    
    def lapl(self, image: np.ndarray) -> np.ndarray:
        # Central differences for the Laplacian (2nd derivatives)
        image_hori_fwd_shift = np.concatenate((image[..., 1:], image[..., :1]), axis=-1)
        image_hori_bwd_shift = np.concatenate((image[..., -1:], image[..., :-1]), axis=-1)
        
        image_vert_fwd_shift = np.concatenate((image[1:, ...], image[:1, ...]), axis=0)
        image_vert_bwd_shift = np.concatenate((image[-1:, ...], image[:-1, ...]), axis=0)
        
        lap_hori = image_hori_fwd_shift - 2 * image + image_hori_bwd_shift
        lap_vert = image_vert_fwd_shift - 2 * image + image_vert_bwd_shift
        return lap_hori + lap_vert


def tykhonov_gradient(noise_image, lam, iterations, diff_init):
    u_t = noise_image
    max_ite = iterations
    while iterations> 1:
        iterations -= 1
        J_d = u_t - noise_image - lam * diff_init.lapl(u_t)
        u_t = u_t - 0.001 * J_d
        if max_ite % iterations == 0:
            print(max_ite % iterations)
            print(f'processing {(max_ite - iterations)/max_ite*100 :.2f}%')
    return u_t


def tykhonov_fourier_denoise(image, lam):
    shape = image.shape
    f = np.fft.fft2(image)
    
    #frequency grids
    i = np.linspace(0, 1, shape[0])
    j = np.linspace(0, 1, shape[1])
    I, J = np.meshgrid(i, j, indexing='ij')

    #the denominator 
    denom = 1 + 8 * lam * (np.square(np.sin(np.pi * I)) + np.square(np.sin(np.pi * J)))
    
    u = f / denom
    
    denoised_image = np.fft.ifft2(u)
    denoised_image = np.real(denoised_image)
    
    return denoised_image


if __name__ == '__main__':
    noise_image = plt.imread('images/cameraman_sp.png')

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

    plt.show()

