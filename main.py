import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 
# from sklearn.metrics import  
from cluster import (KMeans)
from cluster.visualization import plot_3d_clusters
from sklearn.metrics import silhouette_score


def main(): 
    # Set random seed for reproducibility
    np.random.seed(42)
    # Using sklearn iris dataset to train model
    og_iris = np.array(load_iris().data)
    
    # Initialize your KMeans algorithm
    kmeans = KMeans(k=3, metric='euclidean', max_iter=500, tol=1e-4)
    
    
    # Fit model
    kmeans.fit(og_iris)
    

    # Load new dataset
    df = np.array(pd.read_csv('data/iris_extended.csv', 
                usecols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']))

    # Predict based on this new dataset
    predictions = kmeans.predict(df)
    
    
    # You can choose which scoring method you'd like to use here:
    silhouette_avg = silhouette_score(df, predictions)
    
    # Plot your data using plot_3d_clusters in visualization.py
    plot_3d_clusters(df, predictions, kmeans.get_centroids(), silhouette_avg)

    
    # Try different numbers of clusters
    distortions = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans_temp = KMeans(k=k, metric='euclidean', max_iter=500, tol=1e-4)
        kmeans_temp.fit(df)
        distortions.append(kmeans_temp.get_error())


    
    # Plot the elbow plot
    plt.figure(figsize=(8,5))
    plt.plot(K_range, distortions, marker='o')
    plt.title('Elbow Plot k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

    
    # Question: 
    # Please answer in the following docstring how many species of flowers (K) you think there are.
    # Provide a reasoning based on what you have done above: 
    
    """
    How many species of flowers are there: 
    3
    
    Reasoning: 
    The elbow plot shows a significant drop in inertia up to K=3, 
    beyond which the rate of improvement slows down. 
    This indicates the presence of three clusters.
    
    
    
    
    """

    
if __name__ == "__main__":
    main()