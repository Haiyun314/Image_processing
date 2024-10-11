# Image Processing

This repository implements algorithms learned from the lecture "Mathematical Image Processing."

## Overview

In this project, we explore various image processing techniques.


## Requirements

To run this project, you need the following dependencies:

- Python 3.10
- Required Python packages (listed in `requirements.txt`):


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Haiyun314/Image_processing.git

2. You can install the required packages using:
    ```bash
    pip install -r requirements.txt

3. Create a build directory:
    ```bash
    mkdir build
    cd build
4. Run CMake::
    ```bash
    cmake ..

5. Build the project::
    ```bash
    cmake --build . --target RunMain

6. Run Tests:
    ```bash
    cmake --build . --target RunTests

### Usage
To run the algorithms, execute the appropriate Python scripts in the src directory. For example:
    ```bash
    python src/main.py

## Denoising Methods

### Tikhonov Gradient and Tikhonov Fourier Denoise

![Denoising Result](./results/denoising.png)

### Histogram Analysis

![Histogram](./results/histogram.png)

### Filters

![Filter Results](./results/filters.png)

### Upscaling Techniques

![Upscaling Result](./results/upscaling.png)
