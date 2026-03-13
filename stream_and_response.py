import os
import time
import torch
import UdpComms as U
from data_processing import load_and_preprocess
from graph_construction import build_graph  
from autoencoder import GraphAutoencoder  
from clustering import assign_test_samples


folder_to_monit = ".."  #put your folder to monitor here
model_saved = ".."  
centroids_saved = ".." 

#Initialize UDP Communication
sock = U.UdpComms(udpIP="192.168.137.1", portTX=8000, portRX=8001, enableRX=True, suppressWarnings=True)

#Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#Load Centroids
if os.path.exists(centroids_saved):
    centroids = torch.load(centroids_saved)
else:
    raise FileNotFoundError(f"Centroids file not found: {centroids_saved}")

#Process file
def process_file(file_path):
    try:

        processed_data = load_and_preprocess(file_path)

        new_graph = build_graph(processed_data)

        #Extract input dimension from graph data
        input_dim = new_graph.x.shape[-1]  

        #Initialize and load the Autoencoder
        hidden_dim = 256  
        latent_dim = 32   

        model = GraphAutoencoder(input_dim, hidden_dim, latent_dim).to(device)
        model.load_state_dict(torch.load(model_saved))  
        model.eval()

        #Assign_test_samples
        test_sample = [("temp_sample", new_graph)]  

        assigned_cluster_info = assign_test_samples(
            model=model,
            test_data=test_sample,  
            centroids=centroids,
            node_indices=[7, 8],  ##nodes from where check the distance
            stride=5,
            feature_name="Segment Acceleration"
        )

        #Extract assigned cluster and node-wise distances
        assigned_cluster = list(assigned_cluster_info.keys())[0]  
        distances = assigned_cluster_info[assigned_cluster]  

        #extact distances for the chosen nodes
        distance_node_1 = distances.get("Node 7", "N/A")  
        distance_node_2 = distances.get("Node 8", "N/A") 

        #Construct the UDP message
        result_message = (
            f"Your movement has the following distances to the cluster: "
            f"Node 7: Right forearm -> {distance_node_1:.3f}, "
            f"Node 8: Right Hand -> {distance_node_2:.3f}. "
            f"Assigned Cluster: {assigned_cluster}"
        )


        print(result_message)
        sock.SendData(result_message)  #Send to Unity

        os.remove(file_path)  #Delete the file after processing

    except Exception as e:
        print(f"Error processing {file_path}: {e}")



processed_files = set()

while True:
    #Get all current `.xlsx` files
    files = {f for f in os.listdir(folder_to_monit) if f.endswith(".xlsx")}
    new_files = files - processed_files  #Detect new files

    for new_file in new_files:
        file_path = os.path.join(folder_to_monit, new_file)
        
        process_file(file_path)  # Process the new file

        #remove from processed_files if file was deleted
        if not os.path.exists(file_path):
            processed_files.discard(new_file) 

    #Update processed files list
    processed_files.update(new_files)

    time.sleep(1)