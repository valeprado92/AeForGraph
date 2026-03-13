import os
import pandas as pd


#Paths
input_folder = ".."  #put path here
preproc_folder = ".."

#Create output folder
if not os.path.exists(preproc_folder):
    os.makedirs(preproc_folder)

#Process xlsx file
for file_name in os.listdir(input_folder):
    if file_name.endswith(".xlsx") and not file_name.startswith("~$"):  #Avoid temp files (to avoid a pb)
        input_file_path = os.path.join(input_folder, file_name)
        output_file_path = os.path.join(preproc_folder, file_name)
        
        excel_data = pd.ExcelFile(input_file_path)
        writer = pd.ExcelWriter(output_file_path, engine='xlsxwriter')  
        
        for sheet_name in excel_data.sheet_names:
            sheet_data = pd.read_excel(input_file_path, sheet_name=sheet_name)
            
            #Remove first 60 and last 30 frames if the sheet has sufficient rows (removing 1 sec and half sec)
            if len(sheet_data) > 90:
                modified_sheet_data = sheet_data.iloc[70:-60]
            else:
                modified_sheet_data = sheet_data  #If not enough rows keep as is
            
            #Write the modified sheet to the new file
            modified_sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)
        
        writer.close()
        print(f"Processed and saved: {output_file_path}")

