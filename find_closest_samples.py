import torch
import torch.nn.functional as F
import os
import numpy as np
from graph_construction import build_graph



def extract_sliding_windows(sequence, window_size, stride):   
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



def find_closest_samples_to_centroids(model, train_data, output_folder, labeled_files, top_k, stride):
    """
    Finds the closest samples in the training set to each cluster reference sample using Sliding Window technique
    
    Args:
        labeled_files: initial labeled files
        top_k: Number of closest matches to add per cluster
        stride: Step size for sliding windows.

    Returns:
        dict: Updated labeled_files with additional closest samples
    """
    model.to("cuda")
    model.eval()

    labeled_data = {cluster: [] for cluster in labeled_files.keys()}

    #Load labeled data
    for cluster_name, files in labeled_files.items():
        for file in files:
            file_path = os.path.join(output_folder, f"{file}.npy")
            if os.path.exists(file_path):
                try:
                    data = build_graph(np.load(file_path, allow_pickle=True).item())
                    labeled_data[cluster_name].append(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
            else:
                print(f"Error: {file_path} not found!")

    closest_samples = {cluster: [] for cluster in labeled_data.keys()}


    #loop through each cluster
    for cluster_name, graphs in labeled_data.items():
        all_latent_representations = []

        #extract latent representations of the labeled examples
        for data in graphs:
            with torch.no_grad():
                z, _ = model(data.x.to("cuda"), data.edge_index.to("cuda"))
            all_latent_representations.append(z)

        ref_sample = all_latent_representations[0]  #frst sample in cluster as reference
        len_ref_sample = ref_sample.shape[1]

        print(f"Using reference sample length `{len_ref_sample}` for cluster `{cluster_name}`")

        sample_distances = []

        #Loop through all train samples
        for file_name, data in train_data:
            with torch.no_grad():
                z_train, _ = model(data.x.to("cuda"), data.edge_index.to("cuda"))

            len_train_sample = z_train.shape[1]

            #Train sample is > than reference sample --> Sliding Window
            if len_train_sample > len_ref_sample:
                z_windows = extract_sliding_windows(z_train, len_ref_sample, stride)
                distances = torch.norm(z_windows - ref_sample.unsqueeze(0), dim=(1, 2)).mean(dim=1)
                min_distance = distances.min().item()

            #Train sample is = to reference sample --> Direct Comparison
            elif len_train_sample == len_ref_sample:
                min_distance = torch.norm(z_train - ref_sample, dim=(1, 2)).mean().item()

            #Train sample is < than reference sample --> Zero-Padding
            else:
                pad_size = len_ref_sample - len_train_sample
                z_train_padded = F.pad(z_train, (0, 0, 0, pad_size))  

                min_distance = torch.norm(z_train_padded - ref_sample, dim=(1, 2)).mean().item()

            sample_distances.append((file_name, min_distance))

        #select Top-K closest samples
        sorted_samples = sorted(sample_distances, key=lambda x: x[1])[:top_k]
        closest_samples[cluster_name] = [file_name for file_name, _ in sorted_samples]

        #attach to labeled_files
        labeled_files[cluster_name].extend(closest_samples[cluster_name])

        print(f"Updated labeled files for `{cluster_name}`: {labeled_files[cluster_name]}")

    return labeled_files