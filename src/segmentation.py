import numpy as np
import matplotlib.pyplot as plt
import os
from gradient import Diff
from show_image import show_anim, show_images

def chan_vese(image, max_iter=500, lam = 0.2, dt = 0.1):
    lsf = np.ones_like(image) # level set function
    lsf[image < 0.5] = -1
    record = []
    diff = Diff()
    for _ in range(max_iter):
        inside = image[lsf > 0]
        outside = image[lsf <= 0]
        c1 = inside.mean()
        c2 = outside.mean()
        lsf = lsf + dt * (- np.square(image - c1) + np.square(image - c2) + lam * diff.lapl(lsf))  # Chan-Vese Level Set Update
        # phi = np.sign(lsf) * np.maximum(np.abs(lsf), 0.003)
        record.append(lsf)
    return record


root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir))
image_path = os.path.join(root_path, 'images', 'cameraman_nse.png')
# Load and preprocess image
img = plt.imread(image_path)
# print(img)
# Apply Chan-Vese
phi = chan_vese(img)
# show_images(phi[-1], 1, ['result'], 'result')
show_anim(img, phi, save= 1)