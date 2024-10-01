<br />
<p align="center">
  <h1 align="center">PCA for Face Recognition</h1>

  <p align="center">
This project used Principal Component Analysis (PCA), K-Means clustering, and Nearest Neighbors to analyze and predict professor attributes based on a dataset of over 800 Survivor contestants' faces, exploring the link between facial features and traits.</p>

## Table of Contents

* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
* [Evaluation](#evaluation)

## About The Project

# Data

For this project, I worked with image data provided in two folders. The first folder, data/survivor, contains 839 images (each 70x70 pixels) of castaways from the first 46 seasons of Survivor. The filenames follow a format like S##_first_last.png, where ## is the season number. There are some variations due to middle names or multiple last names, and the host, Jeff Probst, is included as S00_Jeff_Probst.png. This Survivor dataset is primarily used for "training."

The second folder, data/professors, includes 5 images (also 70x70 pixels) of the full-time Computer Science faculty at Florida Southern College, following the format first_last.png. These images are used for "testing."

# Steps

This project aims to explore the Survivor castaway dataset using machine learning techniques to answer several unique questions:

Dimensionality Reduction with PCA: Apply Principal Component Analysis (PCA) to the Survivor faces dataset to reduce dimensionality while maintaining at least 90% of the original variance. This step will use the PCA methods from the scikit-learn library.

Identifying the Least "Face-Like" Professor: Determine which professor looks least like a typical face based on the Survivor dataset. Reconstruct each professor's face using a limited number of principal components, then calculate the Euclidean distance between the reconstructed and original faces. The professor with the largest distance is considered least likely to resemble a "face."

Finding the Next Survivor Host: Assess which professor is most likely to be the next host of Survivor. Project each professor’s image into the reduced "Survivor face space" and use nearest neighbor classification to find the professor who looks most similar to Jeff Probst.

Assigning Professors to Survivor Seasons: Use k-means clustering on the PCA-reduced Survivor faces to determine which season each professor would most likely be on. Assign each professor to the nearest cluster and identify their likely season based on the average season of the castaways in that cluster.

Predicting the Most Likely Survivor Winner: Finally, evaluate which professor is most likely to win Survivor. This will be done using quantitative analysis based on the PCA results, with a creative approach to justify the selection.

## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

* [Python 3.12.3](https://www.python.org/downloads/) or higher

## Evaluation

The analysis file in the repository provides a comprehensive and detailed examination of the project’s results.
<!-- If you want to provide some contact details, this is the place to do it -->

<!-- ## Acknowledgements  -->
