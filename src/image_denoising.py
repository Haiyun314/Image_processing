import numpy as np
import matplotlib.pyplot as plt

def gradient(image: np.ndarray):
    """ forward """
    permutation_x = np.concatenate([image[:, -1:], image[:, :-1]], axis= -1)
    permutation_y = np.concatenate([image[-1:, :], image[:-1, :]], axis = 0)
    grad_x = permutation_x - image
    grad_y = permutation_y - image
    return np.stack((grad_x, grad_y), axis= -1)

def lap(grad: np.ndarray):
    """ backward """
    image_x_grad, image_y_grad = grad[..., 0], grad[..., 1]
    permutation_x = np.concatenate([image_x_grad[:, -1:], image_x_grad[:, :-1]], axis= -1)
    permutation_y = np.concatenate([image_y_grad[-1:, :], image_y_grad[:-1, :]], axis= 0)
    grad_xx = image_x_grad - permutation_x
    grad_yy = image_y_grad - permutation_y
    return grad_xx + grad_yy

def ROF_gradient(image: np.ndarray ,lmbd: float, num_iters: int):
    u = image
    t = 0.01
    while num_iters:
        num_iters -= 1
        grad = gradient(u)
        grad_norm = grad/np.sqrt(grad[..., 0]**2 + grad[..., 1]**2 + 0.00001)
        u = u - t * (- lmbd * lap(grad_norm) + u - image)
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
    m, n = image.shape
    u = np.zeros((m, n))
    grad_init = np.zeros((m, n, 2))
    u_bar = u.copy()
    
    L_squared = 8.0
    
    # Ensure that sigma * tau * L_squared < 1
    assert sigma * tau * L_squared < 1, "Step size parameters do not satisfy the convergence condition."
    
    for _ in range(num_iters):
        grad_u_bar = gradient(u_bar)
        
        # proximal operator for total variation
        grad_init = prox_tv(grad_init + sigma * grad_u_bar, lmbd)
        
        # divergence of the grad
        div_p = lap(grad_init)
        u_new = prox_l2(u - tau * div_p, image, tau)

        u_bar = u_new + theta * (u_new - u)
        u = u_new
    
    return u

if __name__ == '__main__':
    img = plt.imread('./images/cameraman_sp.png')
    lmbd = 0.1 
    solution = ROF_gradient(img, 1, 100)
    denoised_image = ROF_primal_dual(img, lmbd)
    print(solution, denoised_image)
