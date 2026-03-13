'''
All the necessary lists and where they are used


'''

#sheets considered in the data_processing function
sheets_considered = ["Segment Velocity", "Segment Acceleration", 
          "Segment Angular Velocity", "Segment Angular Acceleration",
          "Joint Angles ZXY", "Sensor Free Acceleration"] 



#selected_nodes from all the available. Used in graph_construction
nodes = [
    'Pelvis', 'L3', 'T8', 'Neck', 'Head', 'Right Shoulder', 'Right Upper Arm',
    'Right Forearm', 'Right Hand', 'Left Shoulder', 'Left Upper Arm',
    'Left Forearm', 'Left Hand', 'Right Upper Leg', 'Right Lower Leg',
    'Right Foot', 'Left Upper Leg', 'Left Lower Leg', 'Left Foot'
]

#Define all edges for the graph. Used in graph_construction
joint_connections = [
    ('Pelvis', 'L3'), ('L3', 'T8'), ('T8', 'Neck'), ('Neck', 'Head'),
    ('Right Shoulder', 'Right Upper Arm'), ('Right Upper Arm', 'Right Forearm'),
    ('Right Forearm', 'Right Hand'), ('Left Shoulder', 'Left Upper Arm'),
    ('Left Upper Arm', 'Left Forearm'), ('Left Forearm', 'Left Hand'),
    ('Pelvis', 'Right Upper Leg'), ('Right Upper Leg', 'Right Lower Leg'),
    ('Right Lower Leg', 'Right Foot'), ('Pelvis', 'Left Upper Leg'),
    ('Left Upper Leg', 'Left Lower Leg'), ('Left Lower Leg', 'Left Foot'),
    ('T8', 'Right Shoulder'), ('T8', 'Left Shoulder')
]

#node importance values, taken from variance_per_features 
#used in graph_construction
node_importance = {
    "Pelvis": 0.003, "L3": 0.006, "T8": 0.017, "Neck": 0.006, "Head": 0.005,
    "Right Shoulder": 0.072, "Right Upper Arm": 0.096, "Right Forearm": 0.190,
    "Right Hand": 0.184, "Left Shoulder": 0.057, "Left Upper Arm": 0.050,
    "Left Forearm": 0.098, "Left Hand": 0.053, "Right Upper Leg": 0.016,
    "Right Lower Leg": 0.009, "Right Foot": 0.030, "Left Upper Leg": 0.026,
    "Left Lower Leg": 0.031, "Left Foot": 0.051,
}


#selected angles to add as node features. Used in graph_construction
selected_joint_angles = {
    'Right Forearm': "Right Elbow Flexion/Extension",
    'Left Forearm': "Left Elbow Pronation/Supination",
    'Right Lower Leg': "Right Knee Internal/External Rotation",
    'Left Lower Leg': "Left Knee Internal/External Rotation",
    'Right Shoulder': "Right T4 Shoulder Flexion/Extension",
    'Left Shoulder' : "Left T4 Shoulder Flexion/Extension",
    'Right Hand' : "Right Wrist Pronation/Supination",
    'Left Hand' : "Left Wrist Pronation/Supination"
}


# Define the order of features. Used in clustering to extract the valyues for printing test results
FEATURE_NAMES = [
    "Segment Velocity", "Segment Acceleration", "Segment Angular Velocity", "Segment Angular Acceleration"]


#selected nodes to print in the test. Used in clustering
node_indices_for_test = [7, 8, 9]