import pandas as pd
import numpy as np
import os
from config_lists import sheets_considered



def load_and_preprocess(filepath):
    xls = pd.ExcelFile(filepath)
    data = {}
    
    for sheet in sheets_considered:
        df = pd.read_excel(xls, sheet_name=sheet)

        df = df.iloc[::1, :] #here I change the steps to consider

        data[sheet] = df
    
    return data

def batch_preprocess(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.endswith(".xlsx") and not file.startswith("~$"):
            filepath = os.path.join(input_folder, file)
            processed_data = load_and_preprocess(filepath)
            np.save(os.path.join(output_folder, file.replace(".xlsx", ".npy")), processed_data)