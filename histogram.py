import numpy as np
import os
import matplotlib.pyplot as plt


def my_hist(image, nbins: int):
    assert 255 >= nbins >= 1, 'numbers of bins should be larger than 1'
    num_bins = dict()           
    shape = image.shape
    histogram = [np.count_nonzero(image == i) for i in range(256)]
    h = int(np.floor(255/nbins))
    if nbins == 255:
        return histogram, np.array(histogram)/(shape[0] * shape[1])
    else:
        for k in range(nbins):
            try: 
                num_bins[k] = np.sum(histogram[k*h : (k+1) * h])
            except IndexError:
                num_bins[k] = np.sum(histogram[k*h :])
        histogram = np.array(list(num_bins.values()))
        return histogram, histogram/(shape[0]*shape[1])


def linear_transform(image, rectify_range: tuple, target_range:tuple):
    mask = np.logical_and(image <= rectify_range[1], image >= rectify_range[0])
    rectify = mask * rectify_range[0]
    return (image - rectify) / (rectify_range[1] - rectify_range[0]) * (target_range[1] - target_range[0]) + target_range[0]


def gamma_transform(image, gama):
    assert gama > 0, 'gama must be larger than zero, if the image is too dark,'
    'choose gama smaller than 1, if the image is too bright, choose gama larger than 1'
    return np.array(255 * np.power(image/255, gama), np.int32)

def histogram_equalization(image, gray_level, relative_hist):
    shape = image.shape
    eq_image = np.empty(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            eq_image[i, j] = (gray_level) * np.sum(relative_hist[: image[i,j]])
    return eq_image

def cumulative_distribution_function(relative_hist):
    cdf = dict()
    for i in range(len(relative_hist)):
        cdf[i] = np.sum(relative_hist[:i])
    return cdf


if __name__ == '__main__':
    path = os.getcwd() + '/images/' + 'pout.png'
    image = plt.imread(path)
    image = np.array(image[:, :] * 255, np.int32)


    histogram, relative_hist = my_hist(image, 255)
    cdf = cumulative_distribution_function(image, relative_hist)
    hist_eq = histogram_equalization(image, 1, relative_hist)
    linear = linear_transform(image, (50, 150), (0, 255))
    gamma = gamma_transform(image, 0.3)


    _, ax = plt.subplots(2, 2, figsize = (7, 7))
    ax[0, 0].imshow(image, cmap= 'gray')
    ax[0, 0].set_title('original')

    ax[0, 1].plot([i for i in range(len(histogram))], histogram)
    second_axies = ax[0, 1].twinx()
    second_axies.plot(cdf.keys(), cdf.values(), c='red')
    ax[0, 1].set_title('histogram/CDF')

    ax[1, 0].imshow(hist_eq, cmap= 'gray')
    ax[1, 0].set_title('hist_eq')

    ax[1, 1].imshow(gamma, cmap= 'gray')
    ax[1, 1].set_title('gamma')
    plt.savefig('results/histogram.png')
    plt.show()
