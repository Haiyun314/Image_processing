import matplotlib.pyplot as plt
import numpy as np
import os

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
                ax[i, j].imshow(images[k])
                ax[i, j].set_title(names[k])
                ax[i, j].axis('off')
                k += 1
    else:
        # Create a 1-row grid of subplots for fewer images
        fig, ax = plt.subplots(1, number_of_images, figsize=(3 * number_of_images, 3))
        if number_of_images == 1:
            ax.imshow(images)
            ax.set_title(names[0])
            ax.axis('off')
        else:
            for i in range(number_of_images):
                ax[i].imshow(images[i])
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

