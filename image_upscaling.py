import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
import scipy.fft

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
            return -np.pow(x, 2) + 3/4
        elif 1/2 < np.abs(x) < 3/2:
            return 1/2*np.square(np.abs(x) - 3/2)
        else:
            return 0
    
    @staticmethod
    def mitchell_netravali(x):
        if np.abs(x) < 1:
            return 7/6 * np.pow(x, 3) - 2 * np.square(x) + 8/9
        elif 1 <= np.abs(x) < 2:
            return -7/18 * np.pow(np.abs(x), 3) + 2 * np.square(x) - 10/3 * x + 16/9
        else :
            return 0


def resize_copy(image, factor: int):
    shape = image.shape
    target_shape = (shape[0]*factor, shape[1]*factor)
    assert 1 <= factor, 'the factor should be large or equal to one'
    template = np.zeros(shape= target_shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            template[factor*i: (i+1)*factor, j*factor: (j +1)* factor] = image[i, j]
    return template


def i_zeropad(image, factor):
    fft_image = np.fft.fft2(image)
    target_shape = (image.shape[0] * factor, image.shape[1] * factor)

    padded_fft_image = np.zeros(target_shape, dtype=fft_image.dtype)
    padded_fft_image[:image.shape[0]//2, :image.shape[1]//2] = fft_image[:image.shape[0]//2, :image.shape[1]//2]
    
    # Apply inverse FFT to obtain the upscaled image
    upscaled_image = np.fft.ifft2(padded_fft_image).real
    
    # Crop the upscaled image to the desired size
    upscaled_image = upscaled_image[:image.shape[0] * factor, :image.shape[1] * factor]
    
    return upscaled_image


def resize_filter(image, factor: int, filter: str):
    """ unfinished """
    f = Filters
    shape = image.shape
    try: 
        if filter == 'tent':
            scaled_shape = shape + 2 * factor
            new_image = np.zeros(scaled_shape)
            scale_linespace = np.linspace(0, len())


            new_image[0, 0] = image[0, 0]
            new_image[0, -1] = image[0, -1]
            new_image[-1, 0] = image[-1, 0]
            new_image[-1, -1] = image[-1, -1]

            rescal_rate = scaled_shape / shape
            for i in range(len(image)):
                for j in range(len(image[0])):
                    new_image[i, j] = filter((i - (i-1)*rescal_rate) )*image[j, i-1] + filter((i * rescal_rate - i))* image[j, i]
            return new_image
        elif filter == 'bell':
            pass
        elif filter == 'mitchell_netravali':
            pass
    except NameError:
        raise 'please input the correct filter name: tent, bell or mitchell_netravali'


if __name__ == '__main__':
    path = os.getcwd() + '/images/test.png'
    image = plt.imread(path)
    image = image[:, :, 0]
    resized_image = i_zeropad(image, 2)
    plt.subplot(1, 2, 1)
    plt.imshow(resized_image)
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.show()
