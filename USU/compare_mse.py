import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Corrected paths based on file search
path_jara = "data/predikce_jara"
path_pavel = "data/predikce_pavel"
PLOT_DIR = "comparison_plots" # Directory to save plots

# Number of keypoints (kp0 to kp22)
NUM_POINTS = 23

# --- Helper Functions ---

def calculate_mse_for_point(df, k):
    """Calculates Prediction and Baseline MSE for a specific keypoint 'k' across all rows in the df."""
    target_x_col = f'skutecna_kp{k}_x'
    target_y_col = f'skutecna_kp{k}_y'
    pred_x_col = f'predikce_kp{k}_x'
    pred_y_col = f'predikce_kp{k}_y'
    base_x_col = f'nejvetsi_vaha_kp{k}_x'
    base_y_col = f'nejvetsi_vaha_kp{k}_y'

    mse_pred_x, mse_pred_y = np.nan, np.nan
    mse_base_x, mse_base_y = np.nan, np.nan

    # Check if prediction columns exist
    pred_cols = [target_x_col, target_y_col, pred_x_col, pred_y_col]
    if not all(col in df.columns for col in pred_cols):
        # print(f"    - Point {k}: Missing prediction columns, skipping prediction MSE.")
        pass # Keep prediction MSEs as NaN
    else:
        try:
            df_pred = df[pred_cols].apply(pd.to_numeric, errors='coerce').dropna()
            if not df_pred.empty:
                sq_err_pred_x = (df_pred[target_x_col] - df_pred[pred_x_col])**2
                sq_err_pred_y = (df_pred[target_y_col] - df_pred[pred_y_col])**2
                mse_pred_x = np.mean(sq_err_pred_x)
                mse_pred_y = np.mean(sq_err_pred_y)
        except Exception as e:
            print(f"    - Point {k}: Error during prediction MSE calculation: {e}")
            # Keep prediction MSEs as NaN
            
    # Check if baseline columns exist
    base_cols = [target_x_col, target_y_col, base_x_col, base_y_col]
    if not all(col in df.columns for col in base_cols):
        # print(f"    - Point {k}: Missing baseline columns, skipping baseline MSE.")
         pass # Keep baseline MSEs as NaN
    else:
        try:
            df_base = df[base_cols].apply(pd.to_numeric, errors='coerce').dropna()
            if not df_base.empty:
                sq_err_base_x = (df_base[target_x_col] - df_base[base_x_col])**2
                sq_err_base_y = (df_base[target_y_col] - df_base[base_y_col])**2
                mse_base_x = np.mean(sq_err_base_x)
                mse_base_y = np.mean(sq_err_base_y)
        except Exception as e:
            print(f"    - Point {k}: Error during baseline MSE calculation: {e}")
            # Keep baseline MSEs as NaN

    return mse_pred_x, mse_pred_y, mse_base_x, mse_base_y

def process_directory(dir_path, calculate_baseline=False):
    """Processes all CSV files, calculates average per-point prediction (and optionally baseline) MSEs."""
    point_mses_pred_x = {i: [] for i in range(NUM_POINTS)}
    point_mses_pred_y = {i: [] for i in range(NUM_POINTS)}
    point_mses_base_x = {i: [] for i in range(NUM_POINTS)} if calculate_baseline else None
    point_mses_base_y = {i: [] for i in range(NUM_POINTS)} if calculate_baseline else None
    processed_files_count = 0

    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found - {dir_path}")
        return None, None, None, None

    print(f"Processing directory: {dir_path}")
    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(dir_path, filename)
            try:
                df = pd.read_csv(file_path)
                file_processed = False
                for k in range(NUM_POINTS):
                    mse_pred_x, mse_pred_y, mse_base_x, mse_base_y = calculate_mse_for_point(df, k)
                    
                    if not np.isnan(mse_pred_x):
                        point_mses_pred_x[k].append(mse_pred_x)
                        file_processed = True
                    if not np.isnan(mse_pred_y):
                        point_mses_pred_y[k].append(mse_pred_y)
                        file_processed = True
                        
                    if calculate_baseline:
                        if not np.isnan(mse_base_x):
                            point_mses_base_x[k].append(mse_base_x)
                            # No need to mark file_processed again
                        if not np.isnan(mse_base_y):
                            point_mses_base_y[k].append(mse_base_y)
                            # No need to mark file_processed again
                
                if file_processed:
                    processed_files_count += 1

            except Exception as e:
                print(f"  Error reading/processing file {filename}: {e}")

    if processed_files_count == 0:
         print(f"Warning: No CSV files successfully processed in directory {dir_path}")
         return None, None, None, None

    avg_pred_mse_x = {k: np.nanmean(point_mses_pred_x[k]) if point_mses_pred_x[k] else np.nan for k in range(NUM_POINTS)}
    avg_pred_mse_y = {k: np.nanmean(point_mses_pred_y[k]) if point_mses_pred_y[k] else np.nan for k in range(NUM_POINTS)}
    avg_base_mse_x = None
    avg_base_mse_y = None
    if calculate_baseline:
        avg_base_mse_x = {k: np.nanmean(point_mses_base_x[k]) if point_mses_base_x[k] else np.nan for k in range(NUM_POINTS)}
        avg_base_mse_y = {k: np.nanmean(point_mses_base_y[k]) if point_mses_base_y[k] else np.nan for k in range(NUM_POINTS)}

    print(f"Finished processing {dir_path}. Processed {processed_files_count} files.")
    return avg_pred_mse_x, avg_pred_mse_y, avg_base_mse_x, avg_base_mse_y

# --- Main Execution ---
os.makedirs(PLOT_DIR, exist_ok=True)
print(f"Plots will be saved in: {PLOT_DIR}/")

# Process Jara's data and calculate baseline MSEs here
print("Processing Jara's data (including baseline)...")
jara_pred_mse_x, jara_pred_mse_y, base_mse_x, base_mse_y = process_directory(path_jara, calculate_baseline=True)

# Process Pavel's data (prediction only)
print("Processing Pavel's data...")
pavel_pred_mse_x, pavel_pred_mse_y, _, _ = process_directory(path_pavel, calculate_baseline=False)

# Check if processing was successful
if jara_pred_mse_x is None or pavel_pred_mse_x is None or base_mse_x is None:
    print("Error: Could not calculate MSE. Exiting.")
    exit()

# --- Comparison ---
# Prediction MSEs
jara_vals_pred_x = list(jara_pred_mse_x.values())
jara_vals_pred_y = list(jara_pred_mse_y.values())
pavel_vals_pred_x = list(pavel_pred_mse_x.values())
pavel_vals_pred_y = list(pavel_pred_mse_y.values())

jara_overall_pred_mse_x = np.nanmean(jara_vals_pred_x)
jara_overall_pred_mse_y = np.nanmean(jara_vals_pred_y)
jara_overall_pred_total_mse = jara_overall_pred_mse_x + jara_overall_pred_mse_y

pavel_overall_pred_mse_x = np.nanmean(pavel_vals_pred_x)
pavel_overall_pred_mse_y = np.nanmean(pavel_vals_pred_y)
pavel_overall_pred_total_mse = pavel_overall_pred_mse_x + pavel_overall_pred_mse_y

# Baseline MSEs
base_vals_x = list(base_mse_x.values())
base_vals_y = list(base_mse_y.values())
overall_base_mse_x = np.nanmean(base_vals_x)
overall_base_mse_y = np.nanmean(base_vals_y)
overall_base_total_mse = overall_base_mse_x + overall_base_mse_y

print("--- Overall Average Prediction MSE Results ---")
print(f"Jara    - Prediction Total MSE: {jara_overall_pred_total_mse:.4f} (X: {jara_overall_pred_mse_x:.4f}, Y: {jara_overall_pred_mse_y:.4f})")
print(f"Pavel   - Prediction Total MSE: {pavel_overall_pred_total_mse:.4f} (X: {pavel_overall_pred_mse_x:.4f}, Y: {pavel_overall_pred_mse_y:.4f})")
print(f"Baseline - Total MSE:          {overall_base_total_mse:.4f} (X: {overall_base_mse_x:.4f}, Y: {overall_base_mse_y:.4f})")

if np.isnan(jara_overall_pred_total_mse) or np.isnan(pavel_overall_pred_total_mse):
    print("Warning: Could not calculate overall MSE for comparison due to missing data.")
elif jara_overall_pred_total_mse < pavel_overall_pred_total_mse:
    print("Jara's model has a lower overall average prediction MSE.")
elif pavel_overall_pred_total_mse < jara_overall_pred_total_mse:
    print("Pavel's model has a lower overall average prediction MSE.")
else:
    print("Both models have the same overall average prediction MSE (or comparison failed due to NaNs).")

# --- Visualization ---
points = list(range(NUM_POINTS))

# Data Prep for plotting
jara_plot_x = [jara_pred_mse_x.get(i, np.nan) for i in points]
jara_plot_y = [jara_pred_mse_y.get(i, np.nan) for i in points]
pavel_plot_x = [pavel_pred_mse_x.get(i, np.nan) for i in points]
pavel_plot_y = [pavel_pred_mse_y.get(i, np.nan) for i in points]
base_plot_x = [base_mse_x.get(i, np.nan) for i in points]
base_plot_y = [base_mse_y.get(i, np.nan) for i in points]

# Clean data for box plots
jara_clean_x = [x for x in jara_plot_x if not np.isnan(x)]
jara_clean_y = [y for y in jara_plot_y if not np.isnan(y)]
pavel_clean_x = [x for x in pavel_plot_x if not np.isnan(x)]
pavel_clean_y = [y for y in pavel_plot_y if not np.isnan(y)]
# Baseline clean data (optional for boxplot, but good practice)
base_clean_x = [x for x in base_plot_x if not np.isnan(x)]
base_clean_y = [y for y in base_plot_y if not np.isnan(y)]

# Calculate differences (Jara vs Pavel)
diff_x = [p - j if not (np.isnan(p) or np.isnan(j)) else np.nan for p, j in zip(pavel_plot_x, jara_plot_x)]
diff_y = [p - j if not (np.isnan(p) or np.isnan(j)) else np.nan for p, j in zip(pavel_plot_y, jara_plot_y)]

# Plot 1: Per-Point MSE Comparison (Line Plot - INCLUDING BASELINE)
fig1, ax1 = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
# Plot X MSE
ax1[0].plot(points, jara_plot_x, marker='o', linestyle='-', label='Jara Pred X', alpha=0.8)
ax1[0].plot(points, pavel_plot_x, marker='x', linestyle='--', label='Pavel Pred X', alpha=0.8)
ax1[0].plot(points, base_plot_x, marker='s', linestyle=':', label='Baseline X', color='grey', alpha=0.7)
ax1[0].set_ylabel('Average MSE X')
ax1[0].set_title('Per-Point MSE Comparison (Prediction vs Baseline)')
ax1[0].legend()
ax1[0].grid(True, which='both', linestyle='--', linewidth=0.5)
ax1[0].set_xticks(points)
# Plot Y MSE
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

# Plot 2: Overall MSE Comparison (Bar Chart - INCLUDING BASELINE)
labels = ['Jara', 'Pavel', 'Baseline']
overall_pred_x = [jara_overall_pred_mse_x, pavel_overall_pred_mse_x, overall_base_mse_x]
overall_pred_y = [jara_overall_pred_mse_y, pavel_overall_pred_mse_y, overall_base_mse_y]
overall_pred_total = [jara_overall_pred_total_mse, pavel_overall_pred_total_mse, overall_base_total_mse]

x_pos = np.arange(len(labels))  # the label locations [0, 1, 2]
width = 0.25  # the width of the bars

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

# Plot 3: Distribution of Per-Point Prediction MSEs (Box Plot - Jara vs Pavel only)
fig3, ax3 = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Box plot for MSE X
if jara_clean_x or pavel_clean_x:
    ax3[0].boxplot([jara_clean_x, pavel_clean_x], labels=['Jara X', 'Pavel X'], showmeans=True)
    ax3[0].set_title('Distribution of Per-Point Prediction MSE X')
    ax3[0].set_ylabel('Per-Point Prediction MSE')
    ax3[0].grid(True, linestyle='--', linewidth=0.5)
else:
    ax3[0].text(0.5, 0.5, 'No valid X data for boxplot', ha='center', va='center')

# Box plot for MSE Y
if jara_clean_y or pavel_clean_y:
    ax3[1].boxplot([jara_clean_y, pavel_clean_y], labels=['Jara Y', 'Pavel Y'], showmeans=True)
    ax3[1].set_title('Distribution of Per-Point Prediction MSE Y')
    ax3[1].grid(True, linestyle='--', linewidth=0.5)
else:
    ax3[1].text(0.5, 0.5, 'No valid Y data for boxplot', ha='center', va='center')

fig3.tight_layout()
plot3_path = os.path.join(PLOT_DIR, 'prediction_mse_distribution_comparison.png') # Renamed file
plt.savefig(plot3_path)
print(f"Saved Prediction MSE distribution plot to {plot3_path}")
plt.close(fig3)

# Plot 4: Difference in Per-Point Prediction MSEs (Pavel - Jara)
fig4, ax4 = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

ax4[0].plot(points, diff_x, marker='.', linestyle='-', label='MSE X Difference (Pavel - Jara)')
ax4[0].axhline(0, color='grey', linestyle='--', linewidth=0.8)
ax4[0].set_ylabel('Prediction MSE Difference')
ax4[0].set_title('Difference in Per-Point Prediction MSE (Pavel - Jara)')
ax4[0].legend()
ax4[0].grid(True, which='both', linestyle='--', linewidth=0.5)
ax4[0].set_xticks(points)

ax4[1].plot(points, diff_y, marker='.', linestyle='-', label='MSE Y Difference (Pavel - Jara)')
ax4[1].axhline(0, color='grey', linestyle='--', linewidth=0.8)
ax4[1].set_xlabel(f'Point Index (0-{NUM_POINTS-1})')
ax4[1].set_ylabel('Prediction MSE Difference')
ax4[1].legend()
ax4[1].grid(True, which='both', linestyle='--', linewidth=0.5)

fig4.tight_layout(rect=[0, 0.03, 1, 0.98])
plot4_path = os.path.join(PLOT_DIR, 'prediction_mse_difference_per_point.png') # Renamed file
plt.savefig(plot4_path)
print(f"Saved Prediction MSE difference plot to {plot4_path}")
plt.close(fig4)

# plt.show() # Optional