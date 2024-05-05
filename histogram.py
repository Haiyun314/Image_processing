import numpy as np
import os
import matplotlib.pyplot as plt


def my_hist(image, nbins: int, relative_hist: bool):
    assert 255 >= nbins >= 1, 'numbers of bins should be larger than 1'
    num_bins = dict()           
    shape = image.shape
    histogram = [np.count_nonzero(image == i) for i in range(256)]
    h = int(np.floor(255/nbins))
    for k in range(nbins):
        try: 
            num_bins[k] = np.sum(histogram[k*h : (k+1) * h])
        except:
            num_bins[k] = np.sum(histogram[k*h :])
    if relative_hist:
        return num_bins.values() / (shape[0] * shape[1])
    else:
        return num_bins

def linear_transform(image, rectify_range: tuple, target_range:tuple):
    mask_upper_bound = image <= rectify_range[1]
    mask_lower_bound = image >= rectify_range[0]
    mask = np.logical_and(mask_lower_bound, mask_upper_bound)
    rectify = mask * rectify_range[0]
    return (image - rectify) / (rectify_range[1] - rectify_range[0]) * (target_range[1] - target_range[0]) + target_range[0]


def gamma_transform(image, gama):
    assert gama > 0, 'gama must be larger than zero, if the image is too dark,'
    'choose gama smaller than 1, if the image is too bright, choose gama larger than 1'
    return np.array(255 * np.power(image/255, gama), np.int32)

def histogram_equalization(image, relative_hist, gray_level):
    shape = image.shape()
    for i in range(shape[0]):
        for j in range(shape[1]):
            image[i, j] = (gray_level) * np.sum(relative_hist.values()[: image[i,j]])
    return image

def cumulative_distribution_function(relative_hist: dict):
    cdf = dict()
    for i in range(len(relative_hist)):
        cdf[i] = np.sum(relative_hist.values()[:i])
    return cdf


if __name__ == '__main__':
    path = os.getcwd() + '/images/' + 'test.png'
    image = plt.imread(path)
    image = np.array(image[:, :, 0] * 255, np.int32)


    histogram = my_hist(image, 255,0)
    linear = linear_transform(image, (50, 150), (0, 255))
    gamma = gamma_transform(image, 0.3)


    _, ax = plt.subplots(2, 2)
    ax[0, 0].plot(histogram.keys(), histogram.values())
    ax[0, 1].imshow(image, cmap= 'gray')
    ax[1, 0].set_title('linear_transform')
    ax[1, 0].imshow(linear)
    ax[1, 1].imshow(gamma)
    plt.show()
