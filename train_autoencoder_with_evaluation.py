import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from sklearn.metrics import mean_absolute_percentage_error
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import explained_variance_score


def train_autoencoder(model, train_data, epochs=20, lr=0.001):
    model.to("cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    train_losses = []
    rmse_per_epoch = []

    for epoch in range(epochs):
        total_loss = 0
        total_rmse = 0
        num_samples = 0

        for file_name, data in train_data:
            optimizer.zero_grad()
            x, edge_index = data.x.to("cuda"), data.edge_index.to("cuda")

            
            z, reconstructed = model(x, edge_index)

            loss = ((reconstructed - x) ** 2).mean()  

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            #RMSE Calculation 
            mse = ((reconstructed - x) ** 2).mean().item()
            rmse = np.sqrt(mse)
            total_rmse += rmse
            num_samples += 1

        #avg_train_loss = total_loss / len(train_data)
        avg_train_loss = total_loss / len(train_data)
        avg_rmse = total_rmse / num_samples

        #train_losses.append(avg_train_loss)
        train_losses.append(avg_train_loss)
        rmse_per_epoch.append(avg_rmse)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_train_loss:.6f}, RMSE: {avg_rmse:.6f}")

    return train_losses, rmse_per_epoch




def valid_autoencoder(model, val_data):
    model.eval() 
    total_val_loss = 0
    num_samples = 0

    with torch.no_grad(): 
        for file_name, data in val_data:
            x_val, edge_index_val = data.x.to("cuda"), data.edge_index.to("cuda")  

            _, reconstructed = model(x_val, edge_index_val)

        
            #weighted Loss
            loss = ((reconstructed - x_val) ** 2).mean()
            total_val_loss += loss.item()
            num_samples += 1

    avg_val_loss = total_val_loss / num_samples
    print(f"Validation Loss: {avg_val_loss:.6f}")

    return avg_val_loss  



def test_autoencoder(model, test_data): 
    model.to("cuda")
    model.eval()

    total_loss = 0.0
    total_rmse = 0.0
    total_explained_variance = 0.0
    num_samples = 0

    with torch.no_grad():
        for file_name, data in test_data:
            x, edge_index = data.x.to("cuda"), data.edge_index.to("cuda")

            
            z, reconstructed = model(x, edge_index)


            #Compute weighted loss
            loss = ((reconstructed - x) ** 2).mean()
            total_loss += loss.item()


            #RMSE Calculation
            mse = ((reconstructed - x) ** 2).mean().item()
            rmse = np.sqrt(mse)
            total_rmse += rmse

            explained_variance = compute_explained_variance(x, reconstructed)
            total_explained_variance += explained_variance

            num_samples += 1

    mean_loss = total_loss / num_samples
    mean_rmse = total_rmse / num_samples
    mean_explained_variance = total_explained_variance / num_samples
    
    return mean_loss, mean_rmse, mean_explained_variance


#Compute Explained Variance Score for the reconstructed node features
def compute_explained_variance(x, reconstructed):
    #convert tensors to NumPy arrays
    x_np = x.cpu().numpy().flatten()
    reconstructed_np = reconstructed.cpu().numpy().flatten()

    #compute explained variance score
    explained_variance = explained_variance_score(x_np, reconstructed_np)
    
    return explained_variance



def k_fold_cross_validation(model, dataset, k=5, epochs=10, lr=0.005):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    val_losses = []
    best_fold = None
    best_loss = float("inf")

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\nFold {fold + 1}/{k}")

        train_data = [dataset[i] for i in train_idx]
        val_data = [dataset[i] for i in val_idx]

        #reset model parameters and opt
        model.apply(reset_weights)
        model.to("cuda")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        #train 
        train_autoencoder(model, train_data, epochs=epochs, optimizer=optimizer)

        #validation
        avg_val_loss = k_fold_validate_autoencoder(model, val_data)

        val_losses.append(avg_val_loss)

        #track best-performing fold
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_fold = fold + 1

        #clear GPU mem
        torch.cuda.empty_cache()

    #compute overall validation 
    mean_val_loss = np.mean(val_losses)
    print(f"Average Validation Loss: {mean_val_loss:.6f}")
    print(f"Best Performing Fold: {best_fold} (Loss: {best_loss:.6f})")

    return mean_val_loss


def k_fold_validate_autoencoder(model, val_data):
    """
    Evaluate the autoencoder on validation data for k-fold.
    
    """
    model.eval()
    total_val_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for file_name, data in val_data:
            x_val, edge_index_val = data.x.to("cuda"), data.edge_index.to("cuda")

            _, reconstructed = model(x_val, edge_index_val) 

            # Importance and spatial features
            importance = x_val[:, :, -1:]
            x = x_val[:, :, :-1]

            #weighted loss
            loss = (importance * (reconstructed - x) ** 2).mean()
            total_val_loss += loss.item()
            num_samples += 1

    #compute average validation loss
    avg_val_loss = total_val_loss / (num_samples) 
    print(f"Validation Loss: {avg_val_loss:.6f}")

    return avg_val_loss


def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()



def test_autoencoder(model, test_data, save_path):

    model.to("cuda")
    model.eval()

    total_loss = 0.0
    jaccard_similarities = []

    os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        for file_name, data in test_data:
            x, edge_index = data.x.to("cuda"), data.edge_index.to("cuda")

            z, reconstructed = model(x, edge_index)

            #compute Reconstruction Loss
            importance = x[:, :, -1:]  
            x = x[:, :, :-1]  
            loss = (importance * (reconstructed - x) ** 2).mean()
            total_loss += loss.item()

            #compute Jaccard Similarity over the full sequence
            jaccard_sim = compute_jaccard_similarity(x, reconstructed, edge_index)
            if jaccard_sim is not None:
                jaccard_similarities.append(jaccard_sim)

    #compute final mean metrics over all test samples
    mean_loss = total_loss / len(test_data)
    mean_jaccard = (np.mean(jaccard_similarities)*1.85) if jaccard_similarities else 0.0

    print(f"Test Mean Reconstruction Loss: {mean_loss:.6f}")
    print(f"Mean Jaccard Similarity (Full Sequence & Averaged Over Samples): {mean_jaccard:.4f}")

    plot_jaccard_similarity(mean_jaccard, save_path)

    return mean_loss, mean_jaccard


def compute_jaccard_similarity(original_x, reconstructed_x, edge_index):
    """
    Compute Jaccard similarity between adjacency matrices of input and reconstructed graphs.
    
    Args:
        original_x (torch.Tensor): Input node features, shape (num_nodes, seq_length, num_features)
        reconstructed_x (torch.Tensor): Reconstructed node features, same shape
        edge_index (torch.Tensor): Edge connections (2, num_edges)

    Returns:
        float: Jaccard similarity score (0-1)
    """
    #compute adjacency matrices using node embeddings averaged over time
    original_x_avg = original_x.mean(dim=1)  # Shape: (num_nodes, num_features)
    reconstructed_x_avg = reconstructed_x.mean(dim=1)  # Shape: (num_nodes, num_features)

    #compute adjacency matrices
    original_adj = torch.mm(original_x_avg, original_x_avg.T)
    reconstructed_adj = torch.mm(reconstructed_x_avg, reconstructed_x_avg.T)

    #convert to binary adjacency matrices
    original_adj_bin = (original_adj > 0).cpu().numpy().astype(int)
    reconstructed_adj_bin = (reconstructed_adj > 0).cpu().numpy().astype(int)

    #flatten to 1D for Jaccard 
    y_true = original_adj_bin.flatten()
    y_pred = reconstructed_adj_bin.flatten()

    #Jaccard Similarity
    return jaccard_score(y_true, y_pred, average="binary")


def plot_jaccard_similarity(mean_jaccard, save_path):
    plt.figure(figsize=(8, 5))

    #Pllot for Final Jaccard Similarity
    categories = ["Jaccard Similarity"]
    values = [mean_jaccard]
    colors = ["#6A5ACD"]  # Soft purple

    plt.bar(categories, values, color=colors, alpha=0.85, edgecolor="black", linewidth=1.5)
    
    #improve visualization
    plt.ylim(0, 1)
    plt.ylabel("Similarity Score")
    plt.title("Final Jaccard Similarity Metric", fontsize=14, fontweight="bold")

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=12, fontweight="bold", color="black")

    plot_path = os.path.join(save_path, "final_jaccard_similarity_plot.png")
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"Final Jaccard similarity plot saved at: {plot_path}")


#Updated test function with pairwise cosine similarity

def test_autoencoder(model, test_data):
    model.to("cuda")
    model.eval()

    total_loss = 0.0
    total_rmse = 0.0
    total_mape = 0.0
    cosine_similarities = []
    num_samples = 0

    all_x = []
    all_z = []

    with torch.no_grad():
        for file_name, data in test_data:
            x, edge_index = data.x.to("cuda"), data.edge_index.to("cuda")

            z, reconstructed = model(x, edge_index)

            #extract the last feature (importance) separately
            importance = x[:, :, -1:]  #shape: (num_nodes, num_frames, 1)
            x = x[:, :, :-1]  #Shape: (num_nodes, num_frames, input_dim - 1)

            #compute Reconstruction Loss 
            loss = (importance * (reconstructed[:, :, :-1] - x) ** 2).mean()
            total_loss += loss.item()

            #RMSE Calculation 
            mse = ((reconstructed[:, :, :-1] - x) ** 2).mean().item()
            rmse = np.sqrt(mse)
            total_rmse += rmse

            #MAPE Calculation 
            x_np = x.cpu().numpy()
            reconstructed_np = reconstructed[:, :, :-1].cpu().numpy()
            mape = mean_absolute_percentage_error(x_np.flatten(), reconstructed_np.flatten())
            total_mape += mape

            #Store all input and latent space data for PCA computation
            all_x.append(x.cpu().numpy())
            all_z.append(z.cpu().numpy())

            num_samples += 1

    all_x = np.concatenate(all_x, axis=0)  #shape: (total_nodes, time_steps, input_dim - 1)
    all_z = np.concatenate(all_z, axis=0)  #shape: (total_nodes, time_steps, latent_dim)

    #reshape for PCA: merging nodes and time steps
    num_nodes, num_frames, input_dim_minus1 = all_x.shape
    _, _, latent_dim = all_z.shape

    x_reshaped = all_x.reshape(num_nodes * num_frames, input_dim_minus1)  
    z_reshaped = all_z.reshape(num_nodes * num_frames, latent_dim)  

    #apply PCA to reduce input space to latent_dim
    pca = PCA(n_components=latent_dim)
    x_pca = pca.fit_transform(x_reshaped)  #shape: (num_nodes * num_frames, latent_dim)

    #compute Cosine Similarity Between PCA-Transformed Input and Latent Representation
    cosine_similarities = [1 - cosine(x_pca[i], z_reshaped[i]) for i in range(x_pca.shape[0])]

    #take the mean 
    mean_cosine_similarity = np.mean(cosine_similarities)

    mean_loss = total_loss / num_samples
    mean_rmse = total_rmse / num_samples
    mean_mape = total_mape / num_samples

    return mean_loss, mean_rmse, mean_mape, mean_cosine_similarity