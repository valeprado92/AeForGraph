import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


excel_dir = ".." #put path here

#sheets to extract
sheets_to_extract = ["Segment Velocity", "Segment Acceleration", "Segment Angular Velocity", "Segment Angular Acceleration", "Sensor Free Acceleration"]

#dictionary to store data
feature_data = {sheet: [] for sheet in sheets_to_extract}

excel_files = [file for file in os.listdir(excel_dir) if file.endswith(".xlsx") and not file.startswith("~$")]

for file in excel_files:
    file_path = os.path.join(excel_dir, file)
    xls = pd.ExcelFile(file_path)
    for sheet in sheets_to_extract:
        if sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            feature_data[sheet].append(df)


combined_data = {}

for sheet, dfs in feature_data.items():
    if dfs:
        
        merged_df = pd.concat(dfs, axis=0, ignore_index=True)
        
        #compute variance per column
        variance_per_feature = merged_df.iloc[:, 1:].var()  #exclude first column 
        
        combined_data[sheet] = variance_per_feature


variance_df = pd.DataFrame(combined_data)

#group by node, combining X, Y, Z columns
node_variance = {}

for column in variance_df.index:
    match = re.match(r"(.+) (x|y|z)$", column)  #extract node name
    if match:
        node_name = match.group(1)
        if node_name not in node_variance:
            node_variance[node_name] = []
        node_variance[node_name].append(variance_df.loc[column].sum()) 

#Compute final variance per node with Euclidean norm
final_variance = {node: np.sqrt(sum(np.array(var_values) ** 2)) for node, var_values in node_variance.items()}

node_variance_df = pd.DataFrame.from_dict(final_variance, orient='index', columns=['Variance'])

#compute feature importance
node_variance_df['Importance'] = node_variance_df['Variance'] / node_variance_df['Variance'].sum()

#save on a CSV variance and feat importance
variance_csv_path = ".." #put path here
node_variance_df[['Variance']].to_csv(variance_csv_path, index=True)

importance_csv_path = ".." #put path here
node_variance_df[['Importance']].to_csv(importance_csv_path, index=True)


#Variance 
plt.figure(figsize=(12, 6))
node_variance_df['Variance'].plot(kind='bar', color='skyblue')
plt.xlabel("Node")
plt.ylabel("Variance")
plt.title("Variance Per Node")
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

variance_plot_path = ".." #put path here
plt.savefig(variance_plot_path, bbox_inches="tight", dpi=300)
print(f"Saved variance to: {variance_plot_path}")

#ùfeature importance
plt.figure(figsize=(12, 6))
node_variance_df['Importance'].plot(kind='bar', color='orange')
plt.xlabel("Node")
plt.ylabel("Importance (Sum = 1)")
plt.title("Feature Importance Per Node Based on Variance")
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)


importance_plot_path = ".." #put path here
plt.savefig(importance_plot_path, bbox_inches="tight", dpi=300)
print(f"Saved feature importance to: {importance_plot_path}")

# Show Plots
#plt.show()