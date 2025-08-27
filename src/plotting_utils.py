from numpy.linalg import norm  
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from itertools import cycle

def find_dense_directions(tail_empirique, K=8):
    tail_points = tail_empirique

    directions = tail_points / np.linalg.norm(tail_points, axis=1)[:, None]

    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(directions)

    cluster_centers = kmeans.cluster_centers_
    cluster_centers /= np.linalg.norm(cluster_centers, axis=1)[:, None]
    return cluster_centers

def count_points_per_direction(data, directions, cos_threshold=0.7, min_points=20):
    norms = np.linalg.norm(data, axis=1)
    data_norm = data / norms[:, None]
    dim = data.shape[1]
    results = []
    for i, dir_vec in enumerate(directions):
        cos_sim = data_norm.dot(dir_vec)
        count = np.sum(cos_sim >= cos_threshold)
        if count >= min_points:
            results.append({
                "direction": dir_vec*np.sqrt(dim),  
                "count": count
            })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by="count", ascending=False).reset_index(drop=True)
    return df_results

def select_points_near_direction(data, direction, cos_threshold=0.95):
    norms = np.linalg.norm(data, axis=1)
    directions = data / norms[:, None]
    direction_norm = direction / np.linalg.norm(direction)
    cos_sim = directions.dot(direction_norm)
    mask = cos_sim >= cos_threshold
    return data[mask], norms[mask]

