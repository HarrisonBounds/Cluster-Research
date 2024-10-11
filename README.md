
# Image Color Quantization using Advanced K-Means Clustering Algorithms

## Overview

This project implements advanced k-means clustering algorithms for image color quantization. Color quantization reduces the number of distinct colors in an image, which is essential for various applications such as image compression, palette generation, and artistic effects. By leveraging optimized versions of the k-means algorithm, this project aims to achieve efficient and accurate color clustering, even for large and complex images. We explore a relatively unknown clustering algorithm [Jancey K-Means](#jancey-k-means-jkm) and compare its performace to other popular algorithms on the color quantization problem. 

## Features

- **Multiple K-Means Variants**: Implements standard k-means along with optimized versions like Jancey K-Means (JKM), Weighted Jancey K-Means (WJKM), and their accelerated counterparts using Triangle Inequality Elimination (TJKM and TWJKM).

- **Maximin Initialization**: Uses the Maximin algorithm to initialize cluster centers, ensuring a diverse and well-distributed starting point for clustering.

- **Support for Various Color Counts**: Capable of quantizing images to 4, 16, 64, and 256 colors.

- **Batch and Incremental Updates**: Offers both batch and incremental (alpha-adjusted) updates to cluster centers, providing flexibility in balancing convergence speed and accuracy.

- **Performance Optimization**: Utilizes hashing and efficient data structures to manage color tables and accelerate distance calculations.

- **Image I/O**: Reads and writes images in the binary PPM (P6) format, facilitating easy integration with other image processing tools.

## Table of Contents

- [Algorithms](#algorithms)
  - [Batch K-Means](#standard-k-means)
  - [Jancey K-Means (JKM)](#jancey-k-means-jkm)
  - [Weighted Jancey K-Means (WJKM)](#weighted-jancey-k-means-wjkm)
  - [Triangle Inequality Accelerated Algorithms](#triangle-inequality-accelerated-algorithms)
- [Output](#output)
- [Contributions](#contributions)


## Usage

1. **Prepare Images**

   Place all the PPM (P6) format images you wish to quantize in the `images/` directory. Ensure that the images are in binary PPM format.

2. **Run the Program**

   Execute the compiled program. The program processes each image in the `images/` directory, applying color quantization with different k-means variants and color counts.

3. **View Output**

   Quantized images are saved with filenames indicating the method and number of colors used. For example, `out_astro_bodies_jkm12_16.ppm` indicates that the `astro_bodies.ppm` image was quantized using the JKM algorithm with 16 colors.

## Algorithms

### Batch K-Means

The batch k-means algorithm partitions the image's pixels into k clusters based on color similarity. It iteratively updates cluster centers by minimizing the Mean Squared Error (MSE) between pixel colors and their assigned cluster centers.

### Jancey K-Means (JKM)

JKM introduces modifications to the standard k-means to improve convergence speed and accuracy. It adjusts the update rules for cluster centers based on an alpha parameter, allowing for more controlled updates during each iteration.

### Weighted Jancey K-Means (WJKM)

WJKM extends JKM by incorporating pixel weights, which account for the frequency of each color in the image. This weighting ensures that more common colors have a more significant influence on cluster center updates.

### Triangle Inequality Accelerated Algorithms

To further optimize performance, especially for large datasets, Triangle Inequality Elimination techniques are employed:

- **TJKM**: Accelerates JKM by eliminating unnecessary distance calculations using the triangle inequality, reducing computational overhead.

- **TWJKM**: Combines Weighted JKM with Triangle Inequality Acceleration, offering both weighted updates and performance enhancements.

## Output

The program generates quantized images in the binary PPM (P6) format. Output filenames follow the pattern:

    out_<original_filename>_<method>_<k>.ppm

- `<original_filename>`: Name of the input image.
- `<method>`: Indicates the k-means variant used (e.g., `jkm12` for JKM with alpha=1.2).
- `<k>`: Number of colors used in quantization (e.g., 16).

Example:

    out_astro_bodies_jkm12_16.ppm



## Contributions

Emre Celebi
Jordan Maxwell

