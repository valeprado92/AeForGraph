import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

def analyze_joint_differences_anova(file_paths, sheet_name, joints):
    
    dataframes = [pd.read_excel(path, sheet_name=sheet_name) for path in file_paths]

    results = {}

    for joint in joints:
        joint_columns = [col for col in dataframes[0].columns if joint in col]
        anova_results = {}

        for col in joint_columns:
            #gather the same joint angle column across all files
            samples = [df[col].dropna() for df in dataframes]
            if all(len(s) > 1 for s in samples): 
                stat, p_value = f_oneway(*samples)
                anova_results[col] = {"F-statistic": stat, "P-value": p_value}
            else:
                anova_results[col] = {"F-statistic": None, "P-value": None}

        results[joint] = pd.DataFrame(anova_results).T

    return results

def plot_results_anova(results):
    """
    Plot F-statistic and P-value for each joint.

    """
    for joint, result_df in results.items():
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_xlabel("Joint Angles")
        ax1.set_ylabel("F-statistic", color='tab:blue')
        ax1.bar(result_df.index, result_df["F-statistic"], color='tab:blue', alpha=0.6, label="F-statistic")
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        plt.xticks(rotation=45, ha="right")

        ax2 = ax1.twinx()
        ax2.set_ylabel("P-value", color='tab:red')
        ax2.plot(result_df.index, result_df["P-value"], color='tab:red', marker='o', linestyle='dashed', label="P-value")
        ax2.axhline(0.05, color='red', linestyle='dotted', label="Significance threshold (p=0.05)")
        ax2.tick_params(axis='y', labelcolor='tab:red')

        plt.title(f"{joint} Joint Analysis (ANOVA F-statistic & P-value)")
        fig.tight_layout()
        plt.legend(loc="upper right")
        plt.show()


# === Usage ===
file1 = ".." #put your initial files here, the starting files for the centroids
file2 = ".." #put your initial files here, the starting files for the centroids
file3 = ".." #put your initial files here, the starting files for the centroids

sheet = "Joint Angles XZY"
joints_to_analyze = ["Elbow", "Wrist", "Knee", "Shoulder"]

results = analyze_joint_differences_anova([file1, file2, file3], sheet, joints_to_analyze)

for joint, result_df in results.items():
    print(f"\n=== {joint} Joint Analysis (ANOVA) ===")
    print(result_df)

plot_results_anova(results)