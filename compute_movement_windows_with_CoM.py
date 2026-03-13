import pandas as pd
import numpy as np
import os

SHEETS = ["Segment Velocity", "Segment Acceleration",   
          "Segment Angular Velocity", "Segment Angular Acceleration", 
          "Joint Angles ZXY", "Sensor Free Acceleration", "Center of Mass"]

def compute_movement_windows_CoM_division(file_path, output_folder, threshold=0.5, frame_skip=6, min_frames=120):
    """
    Detects movement windows based on acceleration and joint angle changes
    """

    xlsx_data = pd.ExcelFile(file_path)
    sheets_data = {sheet: xlsx_data.parse(sheet) for sheet in SHEETS}

    acc_df = sheets_data["Segment Acceleration"] #Extract the frame column
    frame_numbers = acc_df.iloc[:, 0].values
    frame_offset = frame_numbers[0]
    frame_numbers = frame_numbers - frame_offset

    #Extract Z-Coordinate from the CoM
    com_df = sheets_data["Center of Mass"]
    com_z = com_df["CoM pos z"].to_numpy()
    mean_com_z = np.mean(com_z)

    #Select Acc Columns Based on CoM Z coordinate (select only the height, 0.85 is choose from average height) height of CoM
    if mean_com_z > 0.85:  #standing
        acc_columns = [
            "Right Hand x", "Right Hand y", "Right Hand z",
            "Right Forearm x", "Right Forearm y", "Right Forearm z",
            "Left Hand x", "Left Hand y", "Left Hand z",
            "Left Forearm x", "Left Forearm y", "Left Forearm z",
            "Right Upper Arm x", "Right Upper Arm y", "Right Upper Arm z",
            "Left Upper Arm x", "Left Upper Arm y", "Left Upper Arm z"
        ]
    else:  #bending
        acc_columns = [
            "Right Hand x", "Right Hand y", "Right Hand z",
            "Right Forearm x", "Right Forearm y", "Right Forearm z",
            "Right Upper Leg x", "Right Upper Leg y", "Right Upper Leg z",
            "Right Lower Leg x", "Right Lower Leg y", "Right Lower Leg z"
        ]

    acc_data = acc_df[acc_columns].to_numpy()

    #Compute acceleration magnitude for selected columns and changes with frame skipping
    acc_magnitude = np.linalg.norm(acc_data.reshape(len(acc_data), -1, 3), axis=2).sum(axis=1) 
    acc_diff = np.abs(acc_magnitude[frame_skip:] - acc_magnitude[:-frame_skip])

    #detect movement start and end based on threshold
    start_indices, end_indices = [], []
    in_movement = False 

    for i, score in enumerate(acc_diff):
        frame_idx = frame_numbers[min(i * frame_skip, len(frame_numbers) - 1)]

        if score > threshold and not in_movement:
            start_indices.append(frame_idx)
            in_movement = True
        elif score < threshold and in_movement:
            end_indices.append(frame_idx)
            in_movement = False

    if len(start_indices) > len(end_indices):
        end_indices.append(frame_numbers[-1])

    #saving
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    valid_windows = [(s, e) for s, e in zip(start_indices, end_indices) if (e - s) >= min_frames]

    if len(valid_windows) > 1:
        for idx, (start, end) in enumerate(valid_windows, 1):
            output_filename = f"{base_filename}_{idx}.xlsx"
            output_path = os.path.join(output_folder, output_filename)

            segmented_data = {sheet: df.loc[(df.iloc[:, 0] >= start + frame_offset) & (df.iloc[:, 0] <= end + frame_offset)]
                              for sheet, df in sheets_data.items()}

            with pd.ExcelWriter(output_path) as writer:
                for sheet, df_segment in segmented_data.items():
                    df_segment.to_excel(writer, sheet_name=sheet, index=False)  #save only the selected sheets

            print(f"Saved: {output_filename} (Frames {start + frame_offset} - {end + frame_offset}, Length: {end - start})")
    else:
        output_path = os.path.join(output_folder, f"{base_filename}.xlsx")
        with pd.ExcelWriter(output_path) as writer:
            for sheet, df in sheets_data.items():
                df.to_excel(writer, sheet_name=sheet, index=False)
        
        print(f"no multiple movements were detected, keeping: {base_filename}.xlsx")


def process_xlsx_folder(input_folder, output_folder, threshold=0.01, frame_skip=8, min_frames=90):
    """
    Processes all XLSX files in a folder and extracts movement windows
    """
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".xlsx") and not file.startswith("~$"): #this is necessary for the temp files
            file_path = os.path.join(input_folder, file)
            print(f"Processing: {file_path}")
            try:
                compute_movement_windows_CoM_division(file_path, output_folder, threshold, frame_skip, min_frames)
            except Exception as e:
                print(f"Error processing {file}: {e}")



input_folder = ".." #put path here
output_folder = ".." 

process_xlsx_folder(input_folder, output_folder, threshold=0.5, frame_skip=6, min_frames=90)