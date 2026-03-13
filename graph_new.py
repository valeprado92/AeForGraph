import numpy as np
import torch
from torch_geometric.data import Data
from config_lists import (nodes, 
                          joint_connections, 
                          node_importance, 
                          selected_joint_angles)


def normalize_features(features, min_std=0.01, scale_factor=1):
    
    mean = np.nanmean(features, axis=0)  
    std = np.nanstd(features, axis=0)  
    std = np.maximum(std, min_std)  

    normalized_features = ((features - mean) / std) * scale_factor

    return normalized_features



def normalize_angles_feat(features, mask_value=1e-6, min_std=0.5, scale_factor=0.1):
    """
    Normalizes angle features to handle the absence of some nodes

    Args:
        mask_value: Placeholder for missing angles
        min_std: Minimum standard deviation to prevent excessive scaling.
        scale_factor: Factor to control the final magnitude of normalized values.

    """

    #Create a mask and ignore placeholder values
    valid_mask = features > mask_value
    valid_values = features[valid_mask]

    mean = np.mean(valid_values) if valid_values.size > 0 else 0
    std = np.std(valid_values) if valid_values.size > 0 else 1 

    std = max(std, min_std)

    #normalize valid values
    normalized_features = np.where(valid_mask, ((features - mean) / std) * scale_factor, mask_value)

    return normalized_features



def extract_node_features(data):
    node_features = []
    eps_for_missing = 1e-6  #value for missing angle

    for joint in nodes:
        joint_features = []

        #extract velocity and acceleration
        for sheet in ["Segment Velocity", "Segment Acceleration",
                      "Segment Angular Velocity", "Segment Angular Acceleration",
                      "Sensor Free Acceleration"]:
            joint_x = data[sheet][f"{joint} x"].values
            joint_y = data[sheet][f"{joint} y"].values
            joint_z = data[sheet][f"{joint} z"].values
            joint_features.append(np.stack([joint_x, joint_y, joint_z], axis=1))

        #create Node importance as a feature
        #node_importance_feat = np.full((joint_features[0].shape[0], 1), node_importance[joint])

        #if joint has an associated angle use real value.
        if joint in selected_joint_angles:
            angle_data = data["Joint Angles ZXY"][selected_joint_angles[joint]].values[:, None]  
        else:
            angle_data = np.full((joint_features[0].shape[0], 1), eps_for_missing)

        #all node features together
        joint_features.append(angle_data)  
        #joint_features.append(node_importance_feat)

        #Store all nodes
        node_features.append(np.hstack(joint_features))  #Shape (num_frames, num_features)

    #convert and normalize
    node_features = np.array(node_features)  
    normalized_features = normalize_features(node_features[:, :, :-1]) 
    normalized_angles = normalize_angles_feat(node_features[:, :, -1:])
    node_features = np.concatenate([normalized_features, normalized_angles], axis=-1) 

    return node_features  #Shape (num_nodes, num_frames, num_features)

def build_graph(data):
    edge_index = np.array([[nodes.index(src), nodes.index(dst)] for src, dst in joint_connections]).T
    x = extract_node_features(data)


    return Data(
        x=torch.tensor(x, dtype=torch.float, device="cuda"), 
        edge_index=torch.tensor(edge_index, dtype=torch.long, device="cuda") 
    )