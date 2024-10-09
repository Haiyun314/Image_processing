import numpy as np
import os
import matplotlib.pyplot as plt

class Filters:
    @staticmethod
    def tent(x):
        if np.abs(x) <= 1:
            return 1 - np.abs(x)
        else:
            return 0
    
    @staticmethod
    def bell(x):
        if np.abs(x) <= 1/2:
            return -np.power(x, 2) + 3/4
        elif 1/2 < np.abs(x) < 3/2:
            return 1/2*np.square(np.abs(x) - 3/2)
        else:
            return 0
    
    @staticmethod
    def mitchell_netravali(x):
        if np.abs(x) < 1:
            return 7/6 * np.power(x, 3) - 2 * np.square(x) + 8/9
        elif 1 <= np.abs(x) < 2:
            return -7/18 * np.power(np.abs(x), 3) + 2 * np.square(x) - 10/3 * x + 16/9
        else :
            return 0


def resize_copy(image: np.ndarray, factor: int):
    shape = image.shape
    target_shape = (shape[0]*factor, shape[1]*factor)
    assert 1 <= factor, 'the factor should be large or equal to one'
    template = np.zeros(shape= target_shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            template[factor*i: (i+1)*factor, j*factor: (j +1)* factor] = image[i, j]
    return template


def i_zeropad(image: np.ndarray, factor: int):
    fft_image = np.fft.fft2(image)
    target_shape = (image.shape[0] * factor, image.shape[1] * factor)

    padded_fft_image = np.zeros(target_shape, dtype=fft_image.dtype)
    padded_fft_image[:image.shape[0]//2, :image.shape[1]//2] = fft_image[:image.shape[0]//2, :image.shape[1]//2]
    
    upscaled_image = np.fft.ifft2(padded_fft_image).real
    upscaled_image = upscaled_image[:image.shape[0] * factor, :image.shape[1] * factor]
    
    return upscaled_image


def interpolation_function(image: np.ndarray, x:float, y:float, filter:str):
    if filter == 'bell':
        f = Filters.bell
    elif filter == 'tent':
        f = Filters.tent
    elif filter == 'mitchell_netravali':
        f = Filters.mitchell_netravali
    x_, y_ = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    frac_x, frac_y = x-x_, y-y_
    try:
        inter_p0 = image[x_, y_] * f(1- frac_x) + image[x_ + 1, y_] * f(frac_x)
        inter_p1 = image[x_ , y_+1] * f(1 - frac_x) + image[x_ + 1, y_ + 1] * f(frac_x)
    except IndexError:
        inter_p0 = image[x_, y_] * f(1- frac_x) + image[x_, y_] * f(frac_x)
        inter_p1 = image[x_ ,y_] * f(1 - frac_x) + image[x_, y_] * f(frac_x)
    return inter_p0 * f(1 - frac_y) + inter_p1 * f(frac_y)


def resize_filter(image:np.ndarray, factor: int, filter: str):
    """ unfinished """
    shape = image.shape
    target_shape = (shape[0]*factor, shape[1]*factor)
    target_image = np.empty(target_shape)
    scale = (shape[0]/target_shape[0], shape[1]/target_shape[1])
    for i in range(target_shape[0]):
        for j in range(target_shape[1]):
            target_image[i, j] = interpolation_function(image, i * scale[0], j * scale[1], filter)
    return target_image

if __name__ == '__main__':
    path = os.getcwd() + '/images/cameraman.png'
    image = plt.imread(path)
    resized_image = resize_filter(image, 2,'mitchell_netravali')
    resized_pad = i_zeropad(image, 2)
    resized_copy = resize_copy(image, 2)

    plt.figure(figsize= (7, 7))
    plt.subplot(2, 2, 2)
    plt.imshow(resized_image)
    plt.title('resized_image')
    plt.subplot(2, 2, 1)
    plt.title('orginal')
    plt.imshow(image)
    plt.subplot(2, 2, 3)
    plt.imshow(resized_pad)
    plt.title('resized_pad')
    plt.subplot(2, 2, 4)
    plt.title('resize_copy')
    plt.imshow(resized_copy)
    plt.savefig('results/upscaling.png')
    plt.show()
