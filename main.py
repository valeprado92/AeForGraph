from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from data_processing import batch_preprocess
from graph_construction import build_graph
from autoencoder_with_eval import GraphAutoencoder
from find_closest_samples import find_closest_samples_to_centroids
from data_augmentation import augment_and_expand_data
from train_autoencoder_with_evaluation import (
    train_autoencoder,
    valid_autoencoder,
    #k_fold_cross_validation,
    test_autoencoder
)
from clustering import (
    calculate_cluster_centroids,
    perform_clustering_on_train,
    assign_test_samples
)
from config_lists import node_indices_for_test
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd


# Paths
input_folder = ".." #put path here
output_folder = ".."
dataset = []
model_save = ".."
centroids_save = ".."
image_folder = ".."
save_path = ".."

'''
# Preprocess data
print("Preprocessing data")
batch_preprocess(input_folder, output_folder)
'''


#Build graphs
print("Building graphs from preprocessed data")
for file in os.listdir(output_folder):
    if file.endswith(".npy"):
        file_name = file.replace(".npy", "")
        data = np.load(os.path.join(output_folder, file), allow_pickle=True).item()
        graph = build_graph(data)
        if graph is not None:
            dataset.append((file_name, graph))

if not dataset:
    raise ValueError("No graphs were created")
print(f"Graph construction completed. {len(dataset)} graphs created.\n")



#Initial labeled files
labeled_files = {
    "cazzuola": ["movement-311"],
    "piccone": ["movement-517"],
    "pala": ["movement-171"]
}
labeled_graphs = set(sum(labeled_files.values(), [])) 


#Separate labeled and unlabeled data
labeled_data = {cluster: [] for cluster in labeled_files.keys()}
unlabeled_data = []

for file_name, graph in dataset:
    if file_name in labeled_graphs:
        for cluster, files in labeled_files.items():
            if file_name in files:
                labeled_data[cluster].append(graph)
    else:
        unlabeled_data.append((file_name, graph))

print(f"Labeled data separated: {len(labeled_graphs)} labeled graphs.\n")


#Train-Valid-Test split
def split_dataset(dataset, train_ratio=0.75, val_ratio=0.1, test_ratio=0.15, random_state=42):
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

    train_set, temp_set = train_test_split(dataset, test_size=(1 - train_ratio), random_state=random_state)
    
    #Calculate validation and test size 
    val_size = val_ratio / (val_ratio + test_ratio)
    val_set, test_set = train_test_split(temp_set, test_size=(1 - val_size), random_state=random_state)

    return train_set, val_set, test_set

print("Splitting data into train and test sets")
train_data, val_data, test_data = split_dataset(unlabeled_data, train_ratio=0.75, val_ratio=0.1, test_ratio=0.15, random_state=42)
print(f"Train data size: {len(train_data)} | Valid data size: {len(val_data)} | Test data size: {len(test_data)}\n")


#Augment training data
print("Augmenting training data")
augmented_data = augment_and_expand_data(train_data, num_new_samples=20)
train_data.extend(augmented_data)
print(f"Data augm completed. Total train data size: {len(train_data)}\n")


#Initialize Model
print("Initializing the autoencoder model")
input_dim = dataset[0][1].x.shape[-1]

model = GraphAutoencoder(input_dim=input_dim, hidden_dim=128, latent_dim=12).to("cuda")
#epoch_train_losses, epoch_rmse = train_autoencoder(model, train_data, epochs=10, lr=0.005)



# Training and Validation Loop
train_losses = []
val_losses = []
rmse_per_epoch = []

for epoch in range(30):  
    epoch_train_losses, epoch_rmse = train_autoencoder(model, train_data, epochs=1, lr=0.005)
    train_losses.extend(epoch_train_losses)
    rmse_per_epoch.extend(epoch_rmse)

    epoch_val_loss = valid_autoencoder(model, val_data)
    val_losses.append(epoch_val_loss)

mean_test_loss, mean_test_rmse, test_explained_variance = test_autoencoder(model, test_data)

torch.save(model.state_dict(), model_save)
print(f"Model saved")

print(f"Test Explained Variance: {test_explained_variance:.6f}")


#Unweighted Loss vs Validation Loss
plt.figure(figsize=(12, 5))
plt.plot(train_losses, label="Train Loss", color='darkblue')
plt.plot(val_losses, label="Validation Loss", color='goldenrod')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Validation Loss")
plt.legend()
plt.savefig(os.path.join(save_path, "train_vs_validation_loss.png"))
plt.close()

#RMSE Over Epochs
plt.figure(figsize=(12, 5))
plt.plot(rmse_per_epoch, label="RMSE per Epoch", color='darkmagenta')
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.title("RMSE Over Training")
plt.legend()
plt.savefig(os.path.join(save_path, "rmse_plot.png"))
plt.close()


#creating a DataFrame to store
data = {
    "Dataset": ["Train", "Validation", "Test"],
    "Loss": [train_losses[-1], val_losses[-1], mean_test_loss],  
    "RMSE": [rmse_per_epoch[-1], np.nan, mean_test_rmse] 
}

df_results = pd.DataFrame(data)
print(df_results)

#save the table 
df_results.to_csv(os.path.join(save_path, "train_valid_test_results.csv"), index=False)

#save the table as an image
plt.figure(figsize=(8, 3))
plt.axis('tight')
plt.axis('off')
table = plt.table(cellText=df_results.values,
                  colLabels=df_results.columns,
                  cellLoc='center',
                  loc='center')
plt.savefig(os.path.join(save_path, "train_valid_test_results.png"))
plt.close()


#Find and add closest samples to the list
print("Finding closest samples to centroids")
labeled_files = find_closest_samples_to_centroids(model, train_data, output_folder, labeled_files, top_k=3, stride=5)


#Calculate centroids
print("Calculating cluster centroids")
centroids = calculate_cluster_centroids(model, labeled_data)#, stride=5)

torch.save(centroids, centroids_save)
print(f"Centroids saved")


#Clustering on training data
print("Assigning training samples to clusters")
train_cluster_labels = perform_clustering_on_train(model, train_data, centroids, stride=5)

#Assign test samples
print("Assigning test samples to clusters")
test_cluster_labels = assign_test_samples(model, test_data, centroids, node_indices_for_test, stride=5, feature_name="Segment Acceleration")  #or others quantities
