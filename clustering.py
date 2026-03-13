import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import os

from config_lists import FEATURE_NAMES



def extract_sliding_windows(sequence, window_size, stride):  #same function used in find_closest_samples
    """
    Extracts sliding windows from a sequence

    Args:
        sequence: sequence, shape (num_nodes, num_frames, num_features)
        window_size: Length of each sliding window
        stride: Step size between windows

    Returns:
        Tensor of extracted windows, shape: (num_windows, num_nodes, window_size, num_features)
    """
    num_frames = sequence.shape[1]
    windows = []

    for start in range(0, num_frames - window_size + 1, stride):
        windows.append(sequence[:, start:start + window_size, :])  #extract window

    return torch.stack(windows, dim=0).to("cuda") if len(windows) > 0 else sequence.unsqueeze(0).to("cuda")



def calculate_cluster_centroids(model, labeled_files, num_medoids=3):
    """
    Calculate multiple medoid centroids for each cluster.

    :param model: The trained autoencoder model
    :param labeled_files: Dictionary of labeled samples per cluster
    :param num_medoids: Number of medoids to select per cluster
    :return: Dictionary with list of medoid tensors for each cluster
    """
    centroids = {}

    for cluster_name, samples in labeled_files.items():
        all_latents = []

        #encode each sample to latent space
        for sample in samples:
            with torch.no_grad():
                z, _ = model(sample.x.to("cuda"), sample.edge_index.to("cuda"))
            all_latents.append(z)

        if not all_latents:
            print(f"Warning: No valid samples for cluster `{cluster_name}`.")
            continue

        #Padding
        max_len = max(z.shape[1] for z in all_latents)
        padded = []
        for z in all_latents:
            pad = max_len - z.shape[1]
            z_padded = F.pad(z, (0, 0, 0, pad))
            padded.append(z_padded)

        padded = torch.stack(padded)  #Shape: (N, nodes, frames, features)
        n = padded.shape[0]

        #pairwise distances
        dist_matrix = torch.zeros((n, n), device=padded.device)
        for i in range(n):
            for j in range(n):
                dist_matrix[i, j] = torch.norm(padded[i] - padded[j])

        #sum distances to others
        total_dist = dist_matrix.sum(dim=1)

        #get indices of the k lowest total distances
        medoid_indices = torch.argsort(total_dist)[:num_medoids].tolist()

        #save list
        cluster_medoids = [padded[idx] for idx in medoid_indices]
        centroids[cluster_name] = cluster_medoids

        print(f"{len(cluster_medoids)} medoids selected for `{cluster_name}`, indices: {medoid_indices}")

    return centroids




def perform_clustering_on_train(model, train_data, centroids, stride=5, centroid_window_size=50, centroid_stride_inside=25):
    """
    Assign each train sample to the closest cluster using ensemble of centroid windows.

    Args:
        centroids: Dictionary with lists of medoid tensors per cluster
        stride: Step size for sliding windows on test samples
        centroid_window_size: Size of windows extracted from centroids
        centroid_stride_inside: Stride to extract windows inside centroids
    """
    cluster_assignments = {}

    def extract_windows_from_centroid(centroid, window_size, stride_inside):
        num_frames = centroid.shape[1]
        windows = []
        for start in range(0, num_frames - window_size + 1, stride_inside):
            window = centroid[:, start:start + window_size, :]
            windows.append(window)
        return windows

    for idx, sample in enumerate(train_data):
        file_name, data = sample

        with torch.no_grad():
            z_train, _ = model(data.x.to("cuda"), data.edge_index.to("cuda"))

        len_train_sample = z_train.shape[1]
        distances = {}

        for cluster_name, medoid_list in centroids.items():
            cluster_window_distances = []

            for centroid in medoid_list:
                #extract windows from centroid
                centroid_windows = extract_windows_from_centroid(centroid, centroid_window_size, centroid_stride_inside)

                for centroid_window in centroid_windows:
                    len_centroid_window = centroid_window.shape[1]

                    if len_train_sample > len_centroid_window:
                        z_windows = extract_sliding_windows(z_train, len_centroid_window, stride)
                        distances_to_centroid = torch.norm(z_windows - centroid_window.unsqueeze(0), dim=(1, 2)).mean(dim=1)
                        min_distance = distances_to_centroid.min().item()

                    elif len_train_sample == len_centroid_window:
                        min_distance = torch.norm(z_train - centroid_window, dim=(1, 2)).mean().item()

                    else:
                        pad_size = len_centroid_window - len_train_sample
                        z_padded = F.pad(z_train, (0, 0, 0, pad_size))
                        min_distance = torch.norm(z_padded - centroid_window, dim=(1, 2)).mean().item()

                    cluster_window_distances.append(min_distance)

            #aggregate: take average distance across all centroid windows
            avg_distance = np.mean(cluster_window_distances)
            distances[cluster_name] = avg_distance

        assigned_cluster = min(distances, key=distances.get)
        cluster_assignments[file_name] = assigned_cluster

    return cluster_assignments



def assign_test_samples(model, test_data, centroids, node_indices, stride=5, feature_name="Segment Acceleration",
                        centroid_window_size=50, centroid_stride_inside=25):
    """
    Assign test samples to the closest cluster using ensemble of centroid windows.

    Args:
        centroids: Dictionary with lists of medoid tensors per cluster
        node_indices: list of node indices for analysis
        stride: Step size for sliding windows on test samples
        feature_name: Feature to plot
        centroid_window_size: Size of windows extracted from centroids
        centroid_stride_inside: Stride to extract windows inside centroids
    """
    model.eval()

    if feature_name not in FEATURE_NAMES:
        raise ValueError(f"Feature '{feature_name}' not found. Available: {FEATURE_NAMES}")
    feature_idx = FEATURE_NAMES.index(feature_name)

    plot_folder = "C:/Users/valer/Desktop/Informatica - Computer Science/Thesis/Project/autoencoder_cluster/images_and_other_useful/features_plot"
    os.makedirs(plot_folder, exist_ok=True)

    def extract_windows_from_centroid(centroid, window_size, stride_inside):
        num_frames = centroid.shape[1]
        windows = []
        for start in range(0, num_frames - window_size + 1, stride_inside):
            window = centroid[:, start:start + window_size, :]
            windows.append(window)
        return windows
    
    cluster_assignments = {}


    for idx, sample in enumerate(test_data):
        total_start_time = time.time()

        file_name, data = sample
        x_test, edge_index_test = data.x.to("cuda"), data.edge_index.to("cuda")

        with torch.no_grad():
            z_test, _ = model(x_test, edge_index_test)

        len_test_sample = z_test.shape[1]
        distances = {}

        for cluster_name, medoid_list in centroids.items():
            cluster_window_distances = []

            for centroid in medoid_list:
                centroid_windows = extract_windows_from_centroid(centroid, centroid_window_size, centroid_stride_inside)

                for centroid_window in centroid_windows:
                    len_centroid_window = centroid_window.shape[1]

                    if len_test_sample > len_centroid_window:
                        z_windows = extract_sliding_windows(z_test, len_centroid_window, stride)
                        distances_to_centroid = torch.norm(z_windows - centroid_window.unsqueeze(0), dim=(1, 2)).mean(dim=1)
                        min_distance = distances_to_centroid.min().item()

                    elif len_test_sample == len_centroid_window:
                        min_distance = torch.norm(z_test - centroid_window, dim=(1, 2)).mean().item()

                    else:
                        pad_size = len_centroid_window - len_test_sample
                        mask = torch.cat([torch.ones(len_test_sample, device=z_test.device), torch.zeros(pad_size, device=z_test.device)])
                        z_padded = F.pad(z_test, (0, 0, 0, pad_size))
                        min_distance = torch.norm((z_padded - centroid_window) * mask.unsqueeze(0).unsqueeze(-1), dim=(1, 2)).mean().item()

                    cluster_window_distances.append(min_distance)

            avg_distance = np.mean(cluster_window_distances)
            distances[cluster_name] = avg_distance

        assigned_cluster = min(distances, key=distances.get)
        cluster_assignments[file_name] = assigned_cluster

        total_time = time.time() - total_start_time

        print(f"Test sample {file_name} assigned to cluster: {assigned_cluster} (distance: {distances[assigned_cluster]:.3f})")
        print(f"Total Processing Time: {total_time:.5f} seconds")

    return cluster_assignments

