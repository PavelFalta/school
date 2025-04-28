import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_jara = "data/predikce_jara"
path_pavel = "data/predikce_pavel"
PLOT_DIR = "comparison_plots"
NUM_POINTS = 23


def calculate_mse_for_point(df, k):
    """Calculates Prediction and Baseline MSE for a specific keypoint 'k'. Assumes columns exist and are numeric."""
    target_x_col = f'skutecna_kp{k}_x'
    target_y_col = f'skutecna_kp{k}_y'
    pred_x_col = f'predikce_kp{k}_x'
    pred_y_col = f'predikce_kp{k}_y'
    base_x_col = f'nejvetsi_vaha_kp{k}_x'
    base_y_col = f'nejvetsi_vaha_kp{k}_y'

    # Direct calculation, assumes columns exist and data is valid
    sq_err_pred_x = (df[target_x_col] - df[pred_x_col])**2
    sq_err_pred_y = (df[target_y_col] - df[pred_y_col])**2
    mse_pred_x = np.mean(sq_err_pred_x)
    mse_pred_y = np.mean(sq_err_pred_y)

    # Baseline calculation, assumes columns exist
    sq_err_base_x = (df[target_x_col] - df[base_x_col])**2
    sq_err_base_y = (df[target_y_col] - df[base_y_col])**2
    mse_base_x = np.mean(sq_err_base_x)
    mse_base_y = np.mean(sq_err_base_y)

    return mse_pred_x, mse_pred_y, mse_base_x, mse_base_y

def process_directory(dir_path, calculate_baseline=False):
    # Initialize lists directly, will compute mean later
    point_mses_pred_x = [[] for _ in range(NUM_POINTS)]
    point_mses_pred_y = [[] for _ in range(NUM_POINTS)]
    point_mses_base_x = [[] for _ in range(NUM_POINTS)] if calculate_baseline else None
    point_mses_base_y = [[] for _ in range(NUM_POINTS)] if calculate_baseline else None
    processed_files_count = 0

    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found - {dir_path}")
        # Return empty results
        zeros = [0.0] * NUM_POINTS
        return zeros, zeros, zeros if calculate_baseline else None, zeros if calculate_baseline else None

    print(f"Processing directory: {dir_path}")
    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(dir_path, filename)
            # Removed try-except for file reading/processing
            df = pd.read_csv(file_path).apply(pd.to_numeric) # Convert all to numeric at once

            for k in range(NUM_POINTS):
                mse_pred_x, mse_pred_y, mse_base_x, mse_base_y = calculate_mse_for_point(df, k)

                point_mses_pred_x[k].append(mse_pred_x)
                point_mses_pred_y[k].append(mse_pred_y)

                if calculate_baseline:
                    point_mses_base_x[k].append(mse_base_x)
                    point_mses_base_y[k].append(mse_base_y)

            processed_files_count += 1

    if processed_files_count == 0:
         print(f"Warning: No CSV files successfully processed in directory {dir_path}")
         zeros = [0.0] * NUM_POINTS
         return zeros, zeros, zeros if calculate_baseline else None, zeros if calculate_baseline else None

    # Calculate means after processing all files
    # Use list comprehension for conciseness
    avg_pred_mse_x = [np.mean(mses) if mses else 0.0 for mses in point_mses_pred_x]
    avg_pred_mse_y = [np.mean(mses) if mses else 0.0 for mses in point_mses_pred_y]
    avg_base_mse_x = None
    avg_base_mse_y = None
    if calculate_baseline:
        avg_base_mse_x = [np.mean(mses) if mses else 0.0 for mses in point_mses_base_x]
        avg_base_mse_y = [np.mean(mses) if mses else 0.0 for mses in point_mses_base_y]

    print(f"Finished processing {dir_path}. Processed {processed_files_count} files.")
    # Return lists directly instead of dicts
    return avg_pred_mse_x, avg_pred_mse_y, avg_base_mse_x, avg_base_mse_y

os.makedirs(PLOT_DIR, exist_ok=True)
print(f"Plots will be saved in: {PLOT_DIR}/")

print("Processing Jara's data (including baseline)...")
jara_pred_mse_x, jara_pred_mse_y, base_mse_x, base_mse_y = process_directory(path_jara, calculate_baseline=True)

print("Processing Pavel's data...")
pavel_pred_mse_x, pavel_pred_mse_y, _, _ = process_directory(path_pavel, calculate_baseline=False)

# Simplified check: Assume data is valid if lists are not empty (process_directory handles empty cases)
if not jara_pred_mse_x or not pavel_pred_mse_x or not base_mse_x:
    print("Error: Could not calculate MSE. Exiting.")
    exit()

# No need to extract values, already lists
jara_overall_pred_mse_x = np.mean(jara_pred_mse_x)
jara_overall_pred_mse_y = np.mean(jara_pred_mse_y)
jara_overall_pred_total_mse = jara_overall_pred_mse_x + jara_overall_pred_mse_y

pavel_overall_pred_mse_x = np.mean(pavel_pred_mse_x)
pavel_overall_pred_mse_y = np.mean(pavel_pred_mse_y)
pavel_overall_pred_total_mse = pavel_overall_pred_mse_x + pavel_overall_pred_mse_y

overall_base_mse_x = np.mean(base_mse_x)
overall_base_mse_y = np.mean(base_mse_y)
overall_base_total_mse = overall_base_mse_x + overall_base_mse_y

print("--- Overall Average Prediction MSE Results ---")
print(f"Jara    - Prediction Total MSE: {jara_overall_pred_total_mse:.4f} (X: {jara_overall_pred_mse_x:.4f}, Y: {jara_overall_pred_mse_y:.4f})")
print(f"Pavel   - Prediction Total MSE: {pavel_overall_pred_total_mse:.4f} (X: {pavel_overall_pred_mse_x:.4f}, Y: {pavel_overall_pred_mse_y:.4f})")
print(f"Baseline - Total MSE:          {overall_base_total_mse:.4f} (X: {overall_base_mse_x:.4f}, Y: {overall_base_mse_y:.4f})")

if jara_overall_pred_total_mse < pavel_overall_pred_total_mse:
    print("Jara's model has a lower overall average prediction MSE.")
elif pavel_overall_pred_total_mse < jara_overall_pred_total_mse:
    print("Pavel's model has a lower overall average prediction MSE.")
else:
    print("Both models have the same overall average prediction MSE.")

points = list(range(NUM_POINTS))

# Data is already in list format, no need for .get or default values
jara_plot_x = jara_pred_mse_x
jara_plot_y = jara_pred_mse_y
pavel_plot_x = pavel_pred_mse_x
pavel_plot_y = pavel_pred_mse_y
base_plot_x = base_mse_x
base_plot_y = base_mse_y

# Plot 1: Per-Point MSE
fig1, ax1 = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
ax1[0].plot(points, jara_plot_x, marker='o', linestyle='-', label='Jara Pred X', alpha=0.8)
ax1[0].plot(points, pavel_plot_x, marker='x', linestyle='--', label='Pavel Pred X', alpha=0.8)
ax1[0].plot(points, base_plot_x, marker='s', linestyle=':', label='Baseline X', color='grey', alpha=0.7)
ax1[0].set_ylabel('Average MSE X')
ax1[0].set_title('Per-Point MSE Comparison (Prediction vs Baseline)')
ax1[0].legend()
ax1[0].grid(True, which='both', linestyle='--', linewidth=0.5)
ax1[0].set_xticks(points)

ax1[1].plot(points, jara_plot_y, marker='o', linestyle='-', label='Jara Pred Y', alpha=0.8)
ax1[1].plot(points, pavel_plot_y, marker='x', linestyle='--', label='Pavel Pred Y', alpha=0.8)
ax1[1].plot(points, base_plot_y, marker='s', linestyle=':', label='Baseline Y', color='grey', alpha=0.7)
ax1[1].set_xlabel(f'Point Index (0-{NUM_POINTS-1})')
ax1[1].set_ylabel('Average MSE Y')
ax1[1].legend()
ax1[1].grid(True, which='both', linestyle='--', linewidth=0.5)
fig1.tight_layout(rect=[0, 0.03, 1, 0.98])
plot1_path = os.path.join(PLOT_DIR, 'per_point_mse_comparison.png')
plt.savefig(plot1_path)
print(f"Saved per-point MSE plot to {plot1_path}")
plt.close(fig1)

# Plot 2: Overall MSE Comparison
labels = ['Jara', 'Pavel', 'Baseline']
overall_pred_x = [jara_overall_pred_mse_x, pavel_overall_pred_mse_x, overall_base_mse_x]
overall_pred_y = [jara_overall_pred_mse_y, pavel_overall_pred_mse_y, overall_base_mse_y]
overall_pred_total = [jara_overall_pred_total_mse, pavel_overall_pred_total_mse, overall_base_total_mse]

x_pos = np.arange(len(labels))
width = 0.25

fig2, ax2 = plt.subplots(figsize=(10, 7))
rects1 = ax2.bar(x_pos - width, overall_pred_x, width, label='Avg MSE X')
rects2 = ax2.bar(x_pos, overall_pred_y, width, label='Avg MSE Y')
rects3 = ax2.bar(x_pos + width, overall_pred_total, width, label='Avg Total MSE')

ax2.set_ylabel('Overall Average MSE')
ax2.set_title('Overall MSE Comparison (Prediction vs Baseline)')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels)
ax2.legend()
ax2.bar_label(rects1, padding=3, fmt='%.2f')
ax2.bar_label(rects2, padding=3, fmt='%.2f')
ax2.bar_label(rects3, padding=3, fmt='%.2f')
fig2.tight_layout()
plot2_path = os.path.join(PLOT_DIR, 'overall_mse_comparison.png')
plt.savefig(plot2_path)
print(f"Saved overall MSE plot to {plot2_path}")
plt.close(fig2)

# Plot 3: MSE Distribution Comparison
fig3, ax3 = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

ax3[0].boxplot([jara_plot_x, pavel_plot_x], labels=['Jara X', 'Pavel X'], showmeans=True)
ax3[0].set_title('Distribution of Per-Point Prediction MSE X')
ax3[0].set_ylabel('Per-Point Prediction MSE')
ax3[0].grid(True, linestyle='--', linewidth=0.5)

ax3[1].boxplot([jara_plot_y, pavel_plot_y], labels=['Jara Y', 'Pavel Y'], showmeans=True)
ax3[1].set_title('Distribution of Per-Point Prediction MSE Y')
ax3[1].grid(True, linestyle='--', linewidth=0.5)

fig3.tight_layout()
plot3_path = os.path.join(PLOT_DIR, 'prediction_mse_distribution_comparison.png')
plt.savefig(plot3_path)
print(f"Saved Prediction MSE distribution plot to {plot3_path}")
plt.close(fig3)
