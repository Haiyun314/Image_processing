import numpy as np
import matplotlib.pyplot as plt
from gradient import Diff
import os
from show_image import show_images

def ROF_gradient(image: np.ndarray ,lmbd: float, num_iters: int):
    u = image
    t = 0.01
    diff = Diff()
    while num_iters:
        num_iters -= 1
        grad = diff.grad(u)
        grad_norm = np.sum(np.sqrt(grad[0, ...]**2 + grad[1, ...]**2 + 0.00001))
        u = u - t * (- lmbd * diff.lapl(image)/grad_norm + u - image)
    return u

def prox_tv(grad: np.ndarray, lmbd: float):
    """ total variation."""
    norm_grad = np.maximum(1.0, np.sqrt(grad[..., 0]**2 + grad[..., 1]**2) / lmbd)
    grad[..., 0] /= norm_grad
    grad[..., 1] /= norm_grad
    return grad


def prox_l2(u: np.ndarray, image: np.ndarray, tau):
    """ the L2 norm."""
    return (u + tau * image) / (1 + tau)

def ROF_primal_dual(image: np.ndarray, lmbd, num_iters=100, tau=0.02, sigma=0.25, theta=1.0):
    """Perform the primal-dual minimization algorithm for the ROF model."""
    # Initializations
    diff = Diff
    m, n = image.shape
    u = np.zeros((m, n))
    grad_init = np.zeros((m, n, 2))
    u_bar = u.copy()
    
    L_squared = 8.0
    
    # Ensure that sigma * tau * L_squared < 1
    assert sigma * tau * L_squared < 1, "Step size parameters do not satisfy the convergence condition."
    
    for _ in range(num_iters):
        grad_u_bar = diff.grad(u_bar)
        
        # proximal operator for total variation
        grad_init = prox_tv(grad_init + sigma * grad_u_bar, lmbd)
        
        u_new = prox_l2(u - tau * diff.lapl(image), image, tau)

        u_bar = u_new + theta * (u_new - u)
        u = u_new
    
    return u

if __name__ == '__main__':
    image_path = os.path.join(os.getcwd(), 'images')
    print(image_path)
    img = plt.imread(os.path.join(image_path, 'cameraman_sp.png'))
    lmbd = 0.1 
    solution = ROF_gradient(img, 1, 100)
    solution = np.array(solution)
    images = np.stack((img, solution), axis= 0)
    # denoised_image = ROF_primal_dual(img, lmbd)
    names = ['orginal', 'solution']
    show_images(images, 2, names, name= 'solution')
