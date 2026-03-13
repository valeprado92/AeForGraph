import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from config_lists import (nodes, 
                          joint_connections, 
                          node_importance, 
                          selected_joint_angles)
import numpy as np


class GraphAutoencoder(nn.Module):
    '''
    #Graph Autoencoder: GCN for spatial dimension and LSTM fot temporal dimension with layer normalization
    
    '''
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GraphAutoencoder, self).__init__()
        #Encoder: Temporal and Spatial 
        self.encoder_gcn = GCNConv(input_dim - 1, hidden_dim)
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        self.temporal_encoder = nn.LSTM(hidden_dim, latent_dim, batch_first=True)
        #self.temporal_norm = nn.LayerNorm(latent_dim)

        #Decoder: Temporal and Spatial
        self.temporal_decoder = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.decoder_gcn = GCNConv(hidden_dim, input_dim - 1)

    def forward(self, x, edge_index):
        """
        Forward pass of the autoencoder
        
        Args:
            x: shape (num_nodes, num_frames, num_features)
            edge_index: edge index for graph connections

        Returns:
            z: latent embeddings (important for next steps)
            reconstructed: reconstructed node features with the last feature preserved
        """

        #separate node importance weight (last feature)
        importance = x[:, :, -1:]  #Shape: (num_nodes, num_frames, 1)
        x_spatial = x[:, :, :-1]   #Shape: (num_nodes, num_frames, input_dim - 1)

        #Encoder: Spatial and Temporal (GCN + LSTM)
        x_weighted = x_spatial * importance  #node importance applied
        z_list = []  
        for t in range(x.shape[1]):  
            h_t = self.encoder_gcn(x_weighted[:, t, :], edge_index)  #spatially
            h_t = self.encoder_norm(h_t)
            z_list.append(h_t.unsqueeze(1))  #temporal dimension

        #LSTM
        z = torch.cat(z_list, dim=1)  
        z, _ = self.temporal_encoder(z)  #Shape: (num_nodes, time_steps, latent_dim)

        #decoder: Temporal and Spatial (LSTM + GCN)
        decoded_frames = []
        for t in range(z.shape[1]): 
            z_t = z[:, t, :].unsqueeze(0)  
            h_t, _ = self.temporal_decoder(z_t)  #temporally
            h_t = self.decoder_norm(h_t.squeeze(0))  
            h_t = self.decoder_gcn(h_t, edge_index)  #spatially
            decoded_frames.append(h_t.unsqueeze(1)) 

        #stack all decoded along the temporal dimension
        reconstructed_spatial = torch.cat(decoded_frames, dim=1)  #Shape: (num_nodes, time_steps, input_dim - 1)

        #reattach the last feature unchanged
        reconstructed = torch.cat([reconstructed_spatial, importance], dim=2)  #shape: (num_nodes, time_steps, input_dim)

        return z, reconstructed
