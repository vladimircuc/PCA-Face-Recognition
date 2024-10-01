''' 
Summary:
This program uses PCA (Principal Component Analysis) and K-Means Clustering to analyze faces of 
Survivor contestants and professors from a computer science department. The tasks include:

1. PCA for Dimensionality Reduction: Reduce the dimensionality of the face images using PCA, ensuring 
   90% of the variance is retained.
2. Face Similarity: Find which professor looks least like a face (largest Euclidean distance after reconstruction).
3. Next Host Prediction: Determine which professor is most likely to be the next host of Survivor, by finding 
   the closest professor to Jeff Probst using nearest neighbor classification.
4. Season Prediction: Use K-Means clustering to group Survivor faces and predict which season each professor 
   would likely be on.
5. Winner Prediction: Based on K-Means clusters and Survivor rankings, predict which professor is most likely 
   to win Survivor.
   
Data:
- Survivor Dataset: 839 images of Survivor contestants.
- Professor Dataset: 5 images of professors.
- Rankings CSV: A CSV file with rankings of Survivor contestants.

Methods used:
- PCA for dimensionality reduction.
- K-Means for clustering.
- Nearest Neighbors for closest match to Jeff Probst.
'''

import argparse
import os
import pdb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from skimage.io import imread_collection, imread
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.util import montage
from collections import defaultdict
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd

# Define file paths for data (Survivors, Professors, Rankings)
ROOT = os.path.dirname(os.path.abspath(__file__)) # Path to source directory of this file
PROFESSORS = survivor_folder = os.path.join(ROOT, 'data/professors')
SURVIVORS = survivor_folder = os.path.join(ROOT, 'data/survivor')
RANKINGS = os.path.join(ROOT, 'data/rankings.csv')
rankings_file = pd.read_csv(RANKINGS)

# Argument parser to enable different options like debugging, showing eigenfaces, and reconstructed faces
parser = argparse.ArgumentParser(description="Apply unsupervised learning methods to the problem of face recognition")
parser.add_argument('--debug', help='use pdb to look into code before exiting program', action='store_true')
parser.add_argument('--eigface', help='show the eigen faces', action='store_true')
parser.add_argument('--reconst', help='show the original faces reconstructed with the PCs', action='store_true')
parser.add_argument('--profs', help='show the profesors faces reconstructed with the PCs', action='store_true')
parser.add_argument('--mean', help='show the mean face reconstructed by the pca', action='store_true')
parser.add_argument('--kacc', help='show the accuracy of different ks for the k mean cluster', action='store_true')
parser.add_argument('--eigper', help='show the varience explained by the top 10 eigenfaces', action='store_true')

def main(args):
    # Load the data
    survivors, professors, survivor_names, prof_names, seasons, rank_map = load_data()

    print("Creating the PCA model...")
    pca = PCA(n_components=0.90)
    pca.fit(survivors)
    transformed_survivors = pca.transform(survivors)
    transformed_professors = pca.transform(professors)

    #retrieving the eigenfaces
    n_eigenfaces = pca.n_components_
    #calculating the variance
    total_variance_explained = np.sum(pca.explained_variance_ratio_) * 100  # Convert to percentage

    # Reconstruct survivor and professor faces from the PCA model
    reconstructed_survivors = pca.inverse_transform(transformed_survivors)
    reconstructed_profs = pca.inverse_transform(transformed_professors)
    print("PCA model created!")
    print()

    #printing the PCA statistics
    print()
    print("="*50)
    print(f"Number of Eigenfaces Used: {n_eigenfaces}")
    print(f"Total Variance Explained by These Eigenfaces: {total_variance_explained:.2f}%")
    print("HINT: the --eigface flag will also display those eigenfaces")
    print("="*50)
    print()


    #show a table of the top 10 eigenfaces' explained varience 
    if args.eigper:
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, 11), pca.explained_variance_ratio_[:10] * 100)  # Top 5 principal components (eigenfaces)
        plt.xlabel('Principal Component (Eigenface)', fontsize=12)
        plt.ylabel('Variance Explained (%)', fontsize=12)
        plt.title('Variance Explained by Top 10 Eigenfaces', fontsize=14)
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.xticks(range(1, 11))
        plt.tight_layout()
        plt.show()


    print("Building the KMeans Cluster Model...")
    # If --kacc is specified, check KMeans accuracy
    if args.kacc:
        kmeans_accuracy(transformed_survivors)
    kmeans = KMeans(n_clusters=20, random_state=4)
    kmeans.fit(transformed_survivors)
    cluster_labels = kmeans.labels_
    print("KMeans Cluster model created")
    print()

    # Find the professor who looks least like a face
    prof_faces_errors = least_like_a_face(survivors, professors, prof_names, args, pca, transformed_survivors, reconstructed_profs)

    # Find the professor most likely to host Survivor
    nn, distance = most_likely_to_host(transformed_survivors, transformed_professors, prof_names)

    # Find the most likely season for each professor
    cluster_seasons, scores = most_likely_seasons(transformed_survivors, transformed_professors, prof_names, seasons, kmeans)

    # Predict the professor most likely to win Survivor
    cluster_rankings, prof_rankings, winnerIndex = most_likely_winner(kmeans, rank_map, survivor_names, transformed_professors, prof_names)

    # If --debug is specified, use pdb for debugging
    if args.debug:
        pdb.set_trace()

def load_data():
    '''Function to load the data, including Survivor images, Professor images, and CSV ranking data'''
    survivor_faces = []
    professor_faces = []
    survivor_names = []
    prof_names = []
    seasons = []
    
    # Load Survivor images
    print("Loading Survivor Faces...")
    survivor_folder = os.path.join(ROOT, 'data/survivor')
    for file in os.listdir(survivor_folder):
        # Process file name to extract survivor name and season
        base_name = os.path.splitext(file)[0]    
        parts = base_name.split('_')
        full_name = f"{parts[1]} {parts[2]}"
        season = parts[0]
        survivor_names.append(full_name)
        seasons.append(season)
        
        # Load, convert to grayscale, resize, and flatten image
        img = imread(os.path.join(survivor_folder, file))
        if img.shape[-1] == 4: # Remove alpha channel if present
            img = img[..., :3]  
        img = rgb2gray(img)  # Convert to grayscale
        img = resize(img, (70, 70))  # Resize all images to the same size
        survivor_faces.append(img.flatten())  # Flatten image into a 1D array

    print("Survivor Faces Uploaded!")
    
    # Load Professor images
    print("Uploading Professor Faces...")
    professor_folder = os.path.join(ROOT, 'data/professors')
    for file in os.listdir(professor_folder):
        # Process file name to extract professor name
        base_name = os.path.splitext(file)[0]    
        parts = base_name.split('_')
        full_name = f"{parts[0]} {parts[1]}"
        prof_names.append(full_name)
        
        # Load, convert to grayscale, resize, and flatten image
        img = imread(os.path.join(professor_folder, file))
        if img.shape[-1] == 4: # Remove alpha channel if present
            img = img[..., :3]
        img = rgb2gray(img)  # Convert to grayscale
        img = resize(img, (70, 70))  # Resize to the same size
        professor_faces.append(img.flatten())  # Flatten image

    print("Professor Faces Uploaded!")
    
    # Load rankings from the CSV file and map survivor names to rankings
    print("Uploading rankings from the CSV file...")
    rank_map = defaultdict(int)
    for index, row in rankings_file.iterrows():
        parts = row.iloc[0].split('_')
        name = f"{parts[1]} {parts[2]}"
        ranking = row.iloc[1]
        rank_map[name] = int(ranking)

    print("Rankings Uploaded!")
    
    # Return numpy arrays for faces, names, seasons, and rankings
    return np.array(survivor_faces), np.array(professor_faces), np.array(survivor_names), np.array(prof_names), np.array(seasons), rank_map

def least_like_a_face(survivors, professors, prof_names, args, pca, transformed_survivors, reconstructed_profs): 
    '''Function to determine which professor looks the least like a face based on Euclidean distance from reconstructed image'''

    # Reconstruct survivor faces and plot if requested
    if args.reconst:
        reconstructed_survivors = pca.inverse_transform(transformed_survivors)
        m = montage(reconstructed_survivors.reshape((reconstructed_survivors.T.shape[1],) + (70, 70)))
        plt.imshow(m, cmap='gray')
        plt.axis('off')
        plt.title('Original Faces Reconstruction')
        plt.show(block=True)

    # Plot eigenfaces if requested
    if args.eigface:
        m = montage(pca.components_.reshape((pca.components_.T.shape[1],) + (70, 70)))
        plt.imshow(m, cmap='gray')
        plt.axis('off')
        plt.title('Eigenfaces')
        plt.show(block=True)
    
    if args.mean:
        img = pca.mean_.reshape(70,70)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title('PCA computed Mean Face')
        plt.show(block=True)

    # Compute Euclidean distance between original and reconstructed professor faces
    errors = []
    for i in range(professors.shape[0]):
        error = np.linalg.norm(professors[i] - reconstructed_profs[i])
        errors.append(error)

    # Plot professor face reconstructions if requested
    if args.profs:
        reconstructed_profs_reshaped = reconstructed_profs.reshape(-1, 70, 70) #reshape the reconstructed profs
        fig, axes = plt.subplots(1, reconstructed_profs_reshaped.shape[0], figsize=(15, 5)) #create a figure with subplots, the reconstructed images
        # Plot each image in a single row and add the corresponding label
        for i, ax in enumerate(axes):
            ax.imshow(reconstructed_profs_reshaped[i], cmap='gray')
            ax.axis('off')  # Hide the axes
            ax.set_title(f"{prof_names[i]} -- ERR: {round(errors[i], 2)}", fontsize=10)  # Set the name for each image
        plt.suptitle('Professors Faces Reconstruction', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    # Find the professor with the largest reconstruction error (least face-like)
    least_face = np.argmax(errors)
    print("***************************************************************************************")
    print(f"* {prof_names[least_face]} looks the least like a real face with an error of {round(errors[least_face], 2)}.")
    print("***************************************************************************************")
    print()

    return errors

def most_likely_to_host(transformed_survivors, transformed_professors, prof_names): 
    '''Function to determine which professor is most likely to host Survivor based on nearest neighbors to Jeff Probst'''
    jeff_img = transformed_survivors[0].reshape(1, -1)  # Use Jeff Probst as the reference image
    nn = NearestNeighbors(n_neighbors=1)  # Find the closest professor to Jeff
    nn.fit(transformed_professors)
    distance, index = nn.kneighbors(jeff_img)  # Get the closest professor
    closest_professor = prof_names[index[0][0]]
    print("***************************************************************************************")
    print(f"* {closest_professor} is most likely to be the next host of Survivor with a distance of {round(distance[0][0], 2)} from Jeff Probst's face.")
    print("***************************************************************************************")
    print()

    return nn, distance

def most_likely_seasons(transformed_survivors, transformed_professors, prof_names, seasons, kmeans):
    '''Function to determine which season each professor is most likely from based on K-Means clusters'''
    cluster_labels = kmeans.labels_  # Get cluster assignments
    cluster_seasons = defaultdict(list)
    
    # Map each cluster to its corresponding seasons
    for i, cluster in enumerate(cluster_labels):
        cluster_seasons[cluster].append(seasons[i])

    # Predict the season for each professor by finding the most common season in their cluster
    scores = []
    print("***************************************************************************************")
    for i, prof in enumerate(transformed_professors):
        pred_cluster = kmeans.predict(prof.reshape(1, -1))[0]  # Predict the cluster for each professor
        season_list = cluster_seasons[pred_cluster]  # Get the seasons corresponding to the predicted cluster

        # Predict the most common season from the cluster
        pred_season, max_nr, total_seasons = predict_season(season_list)

        # Calculate the percentage of faces in the cluster from the predicted season
        score = max_nr / total_seasons
        scores.append(score)

        print(f"* {prof_names[i]} is most likely from season {pred_season[-2:]} with {max_nr} out of {total_seasons} faces in the cluster from that season.")
        print("-----------------------------------------------------------")
    print("***************************************************************************************")

    mean = np.mean(scores)
    print(f"Average confidence in season prediction: {mean * 100:.2f}%")
    print()

    return cluster_seasons, scores

def predict_season(season_list):
    '''Function to predict the most common season in a cluster'''
    unique_seasons = set(season_list)  # Get unique seasons
    predicted_season = None
    max_nr = 0

    # Find the season that occurs the most in the season list
    for season in unique_seasons:
        new_nr = season_list.count(season)
        if new_nr > max_nr:
            max_nr = new_nr
            predicted_season = season

    return predicted_season, max_nr, len(season_list)

def kmeans_accuracy(transformed_survivors):
    '''Function to check the accuracy of K-Means clustering with different numbers of clusters'''
    s_scores = []
    d_scores = []
    k_values = range(10, 200)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=4)
        kmeans.fit(transformed_survivors)

        # Calculate silhouette score for clustering
        silhouette_avg = silhouette_score(transformed_survivors, kmeans.labels_)
        s_scores.append(silhouette_avg)

        # Calculate Davies-Bouldin index for clustering
        db_index = davies_bouldin_score(transformed_survivors, kmeans.labels_)
        d_scores.append(db_index)
        #print(f"Davies-Bouldin Index for k = {k}: {db_index}")

    # Plot silhouette score for different numbers of clusters
    plt.plot(k_values, s_scores, marker='o')
    plt.title("Silhouette Score vs Number of Clusters (the higher the better)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.show()

    plt.plot(k_values, d_scores, marker='o')
    plt.title("Davies Bouldin Score vs Number of Clusters (the lower the better)")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Davies Bouldin Score")
    plt.show()

def most_likely_winner(kmeans, rank_map, survivor_names, transformed_professors, prof_names): 
    '''Function to determine which professor is most likely to win Survivor, based on cluster rankings'''
    cluster_labels = kmeans.labels_
    cluster_rankings = defaultdict(list)

    # Populate the dictionary with rankings based on clusters
    for i, cluster in enumerate(cluster_labels):
        cluster_rankings[cluster].append(rank_map[survivor_names[i]])

    prof_rankings = []

    # Calculate the average ranking for each professor's cluster
    for i, prof in enumerate(transformed_professors):
        pred_cluster = kmeans.predict(prof.reshape(1, -1))[0]
        ranking_list = cluster_rankings[pred_cluster]

        mean = np.mean(ranking_list)
        prof_rankings.append(mean)

    # Find the professor with the lowest average ranking (most likely to win)
    winner = np.argmin(prof_rankings)

    print("***************************************************************************************")
    print(f"* {prof_names[winner]} is most likely to win, placing around {round(prof_rankings[winner])}th place.")
    print("***************************************************************************************")

    return cluster_rankings, prof_rankings, winner

if __name__ == "__main__": 
    main(parser.parse_args())

