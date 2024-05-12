import numpy as np
import os 
import matplotlib.pyplot as plt

class Filters:
    @staticmethod
    def m():
        return 1/25 * np.ones(shape= (5, 5))
    
    @staticmethod
    def g():
        background = np.zeros(shape= (5, 5))
        for i in range(5):
            for j in range(5):
                background[i, j] = np.abs(3 - i) * np.abs(3 - j)
        return 1/81 * background
    
    @staticmethod
    def gamma():
        backgraound = np.zeros(shape=(3, 3))
        backgraound[0, 1] = -1
        backgraound[1, 0] = -1
        backgraound[1, 2] = 1
        backgraound[2, 1] = 1
        return backgraound
    
    @staticmethod
    def delta():
        backgraound = np.zeros(shape=(3, 3))
        backgraound[0, 1] = 1
        backgraound[1, 0] = 1
        backgraound[1, 2] = 1
        backgraound[2, 1] = 1
        backgraound[1, 1] = -4


def image_filter(image, filters: str):
    f = Filters
    try:
        if filters == 'm':
            filter = f.m()
        elif filters == 'g':
            filter = f.g()
        elif filters == 'gamma':
            filter = f.gamma()
        elif filters == 'delta':
            filter = f.delta()
    except NameError:
        raise 'please change the input name: m, g, gama or delta'
    padded_image = np.pad(image, pad_width= filter.shape[0]//2, constant_values= 0)
    template = np.zeros(shape = image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            template[i, j] = np.sum(np.multiply(padded_image[i: i + filter.shape[0], j: j + filter.shape[1]], filter))
    return template

def median_filter(image, size):
    padded_image = np.pad(image, pad_width= size[0]//2, constant_values= 0)
    template = np.zeros(shape= image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            template[i, j] = np.median(padded_image[i: i + size[0], j : j + size[1]])
    return template


if __name__ == '__main__':
    path = os.getcwd() + '/images/cameraman.png'
    image = plt.imread(path)
    image = image[:, :]

    f_image = image_filter(image, 'g')
    f_image_g = image_filter(image, 'gamma')
    m_image = median_filter(image, (3, 3))
    _, axes = plt.subplots(2, 2, figsize = (7, 7))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('original')
    axes[0, 1].imshow(f_image)
    axes[0, 1].set_title('filter_g')
    axes[1, 0].imshow(m_image)
    axes[1, 0].set_title('median_filter')
    axes[1, 1].imshow(f_image_g)
    axes[1, 1].set_title('filter_gamma')
    plt.savefig('./results/filters')
    plt.show()