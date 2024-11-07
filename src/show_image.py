import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation

def show_images(images: np.ndarray, number_of_images: int, names: list, name: str, save=0):
    if number_of_images > 1:
        assert number_of_images == len(images), f'Expected {number_of_images} images, but got {len(images)}'
        assert len(images) <= 4, f'Too many images: {len(images)} (maximum allowed is 4)'

    if number_of_images == 4:
        # Create a 2x2 grid for 4 images
        fig, ax = plt.subplots(2, 2, figsize=(6, 6))
        k = 0
        for i in range(2):
            for j in range(2):
                ax[i, j].imshow(images[k], cmap= 'gray')
                ax[i, j].set_title(names[k])
                ax[i, j].axis('off')
                k += 1
    else:
        # Create a 1-row grid of subplots for fewer images
        fig, ax = plt.subplots(1, number_of_images, figsize=(3 * number_of_images, 3))
        if number_of_images == 1:
            ax.imshow(images, cmap= 'gray')
            ax.set_title(names[0])
            ax.axis('off')
        else:
            for i in range(number_of_images):
                ax[i].imshow(images[i], cmap= 'gray')
                ax[i].set_title(names[i])
                ax[i].axis('off')

    plt.tight_layout()

    if save:
        result_path = os.path.join(os.getcwd(), 'results')
        # Ensure the 'results' directory exists
        os.makedirs(result_path, exist_ok=True)
        # Save the figure in the 'results' directory with the given name
        save_path = os.path.join(result_path, f'{name}')
        plt.savefig(save_path)
        print(f"Image saved to {save_path}")
    else:
        # Display the plot
        plt.show()

def show_anim(image, contour, save: int = 0):
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(6, 3)
    # Show the first image as the background
    ax[0].imshow(image, cmap= 'gray')
    ax[0].set_title('origin')
    ax[1].imshow(image, cmap= 'gray')
    ax[1].contour(contour[-1], levels = [0], colors= 'yellow')
    def update(frame):
        ax[1].cla()
        ax[1].imshow(image, cmap= 'gray')
        ax[1].contour(contour[frame], levels = [0], colors= 'yellow')
        ax[1].set_title('Chan-vese Segmentation')
        return ax

    ani = FuncAnimation(fig, update, frames=[i*10 for i in range(int(len(contour)/10))])
    if save:
        root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), os.pardir, os.pardir))
        image_path = os.path.join(root_path, 'results', 'animation.gif')
        ani.save(image_path, writer='imagemagick')

    plt.show()
