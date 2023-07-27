# Cluster-Research

The implementation of 8 different K-means clustering algorithms contained in 4 functions. Each performs color quantization on a specific image and analyzes the time comparisons between each algorithm. 

Algorithms implemented:
- Unweighted
  - Batch k-means
  - Jancey k-means
  - Batch k-means + TIE
  - Jancey k-means + TIE
- Weighted
  - Batch k-means
  - Jancey k-means
  - Batch k-means + TIE
  - Jancey k-means + TIE

Authors: Harrison Bounds & M. Emre Celebi

Contact: harrison.bounds777@gmail.com or ecelebi@uca.edu

Running main.py:
-Build the cpp file to get an executable
- Takes two command line arguments
  - location of PPM formatted image
  - Number of Clusters (32, 64, 128, 256)
 
Example: ./main.exe 4.2.03.ppm 32
