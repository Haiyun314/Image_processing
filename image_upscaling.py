import numpy as np
import os
import matplotlib.pyplot as plt
import scipy
import scipy.fft

class Filters:
    def tent(x):
        if np.abs(x) <= 1:
            return 1 - np.abs(x)
        else:
            return 0
        
    def bell(x):
        if np.abs(x) <= 1/2:
            return -np.pow(x, 2) + 3/4
        elif 1/2 < np.abs(x) < 3/2:
            return 1/2*np.square(np.abs(x) - 3/2)
        else:
            return 0
    
    def mitchell_netravali(x):
        if np.abs(x) < 1:
            return 7/6 * np.pow(x, 3) - 2 * np.square(x) + 8/9
        elif 1 <= np.abs(x) < 2:
            return -7/18 * np.pow(np.abs(x), 3) + 2 * np.square(x) - 10/3 * x + 16/9
        else :
            return 0


def resize_copy(image, factor: int):
    shape = image.shape()
    assert 0 <= factor <= np.min(shape), 'the rescaling size is too large or the factor should be large or equal to zero'
    rescaling_temp = np.zeros(3 * shape)
    for i in range(3):
        for j in range(3):
            rescaling_temp[i * shape[0] : (i+1) * shape[0], j * shape[1] : (j + 1) * shape[1]] = image
    return rescaling_temp[(shape[0] - factor) : (2 * shape[0] + factor), (shape[1] - factor) : (2 * shape[1] + factor)]


def i_zeropad(image, factor):
    image = scipy.fft(image)
    padded = np.pad(image, pad_width= factor, constant_values= 0)
    return scipy.ifft(padded)

def resize_filter(image, factor: int, filter: str):
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
    plt.imshow(image)
    plt.show()
    print(path)
