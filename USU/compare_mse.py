import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_jara = "data/predikce_jara"
path_pavel = "data/predikce_pavel"
PLOT_DIR = "comparison_plots" # Directory to save plots

# csv colums: skutecna_kp0_y,skutecna_kp0_x,nejvetsi_vaha_kp0_y,nejvetsi_vaha_kp0_x,predikce_kp0_y,predikce_kp0_x
# Expected points (rows) per file: 0 to 23
NUM_POINTS = 24

# --- Helper Functions ---

def calculate_point_sq_error(df):
    """Calculates squared error for predictions for each point (row)."""
    sq_error_x = (df['skutecna_kp0_x'] - df['predikce_kp0_x'])**2
    sq_error_y = (df['skutecna_kp0_y'] - df['predikce_kp0_y'])**2
    return sq_error_x, sq_error_y

def process_directory(dir_path):
    """Processes all CSV files, calculates average per-point prediction MSEs."""
    # Initialize accumulators for squared errors and counts for each point
    point_sq_error_sum_x = {i: 0.0 for i in range(NUM_POINTS)}
    point_sq_error_sum_y = {i: 0.0 for i in range(NUM_POINTS)}
    point_counts = {i: 0 for i in range(NUM_POINTS)}
    processed_files_count = 0

    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found - {dir_path}")
        return None, None # Return None if directory doesn't exist

    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(dir_path, filename)
            try:
                df = pd.read_csv(file_path)

                # Basic validation: Check columns
                required_cols = ['skutecna_kp0_y', 'skutecna_kp0_x',
                                 'predikce_kp0_y', 'predikce_kp0_x']
                # Include baseline columns for validation only if they exist
                optional_baseline_cols = ['nejvetsi_vaha_kp0_y', 'nejvetsi_vaha_kp0_x']
                cols_to_check = required_cols + [col for col in optional_baseline_cols if col in df.columns]

                if not all(col in df.columns for col in required_cols):
                    print(f"Warning: Skipping file {filename} due to missing required prediction/target columns.")
                    continue

                # Convert relevant columns to numeric
                for col in cols_to_check:
                     df[col] = pd.to_numeric(df[col], errors='coerce')

                # Drop rows with NaN in essential columns
                df.dropna(subset=required_cols, inplace=True)

                if df.empty:
                    print(f"Warning: Skipping file {filename} - no valid numeric data after cleaning.")
                    continue

                # Calculate squared errors for this file
                sq_error_x, sq_error_y = calculate_point_sq_error(df)

                # Add to accumulators based on index (assuming index = point ID 0-23)
                for idx in range(len(df)):
                    point_idx = df.index[idx]
                    if 0 <= point_idx < NUM_POINTS:
                        point_sq_error_sum_x[point_idx] += sq_error_x.iloc[idx]
                        point_sq_error_sum_y[point_idx] += sq_error_y.iloc[idx]
                        point_counts[point_idx] += 1
                    else:
                         print(f"Warning: Skipping row with index {point_idx} in file {filename} (out of expected range 0-{NUM_POINTS-1}).")

                processed_files_count += 1

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    if processed_files_count == 0:
         print(f"Warning: No CSV files successfully processed in directory {dir_path}")
         return None, None

    # Calculate average MSE for each point
    avg_mse_x = {i: point_sq_error_sum_x[i] / point_counts[i] if point_counts[i] > 0 else np.nan for i in range(NUM_POINTS)}
    avg_mse_y = {i: point_sq_error_sum_y[i] / point_counts[i] if point_counts[i] > 0 else np.nan for i in range(NUM_POINTS)}

    return avg_mse_x, avg_mse_y

# --- Main Execution ---
# Create plot directory if it doesn't exist
os.makedirs(PLOT_DIR, exist_ok=True)
print(f"Plots will be saved in: {PLOT_DIR}/")

print("Processing Jara's data...")
jara_mse_x, jara_mse_y = process_directory(path_jara)

print("Processing Pavel's data...")
pavel_mse_x, pavel_mse_y = process_directory(path_pavel)

# Check if processing was successful
if jara_mse_x is None or pavel_mse_x is None:
    print("Error: Could not calculate MSE for one or both researchers. Exiting.")
    exit()

# --- Comparison ---
# Calculate overall average total MSE for comparison
# Use np.nanmean to handle potential NaN values if a point had no data
jara_vals_x_list = list(jara_mse_x.values())
jara_vals_y_list = list(jara_mse_y.values())
pavel_vals_x_list = list(pavel_mse_x.values())
pavel_vals_y_list = list(pavel_mse_y.values())

jara_overall_mse_x = np.nanmean(jara_vals_x_list)
jara_overall_mse_y = np.nanmean(jara_vals_y_list)
jara_overall_total_mse = jara_overall_mse_x + jara_overall_mse_y

pavel_overall_mse_x = np.nanmean(pavel_vals_x_list)
pavel_overall_mse_y = np.nanmean(pavel_vals_y_list)
pavel_overall_total_mse = pavel_overall_mse_x + pavel_overall_mse_y

print("--- Overall Average Prediction MSE Results ---")
print(f"Jara - Average Total MSE: {jara_overall_total_mse:.4f} (X: {jara_overall_mse_x:.4f}, Y: {jara_overall_mse_y:.4f})")
print(f"Pavel - Average Total MSE: {pavel_overall_total_mse:.4f} (X: {pavel_overall_mse_x:.4f}, Y: {pavel_overall_mse_y:.4f})")

if jara_overall_total_mse < pavel_overall_total_mse:
    print("Jara's model has a lower overall average prediction MSE.")
elif pavel_overall_total_mse < jara_overall_total_mse:
    print("Pavel's model has a lower overall average prediction MSE.")
else:
    print("Both models have the same overall average prediction MSE.")

# --- Visualization ---
points = list(range(NUM_POINTS))

# Data Prep for plotting (ensure alignment and handle NaNs)
jara_vals_x = [jara_mse_x.get(i, np.nan) for i in points]
jara_vals_y = [jara_mse_y.get(i, np.nan) for i in points]
pavel_vals_x = [pavel_mse_x.get(i, np.nan) for i in points]
pavel_vals_y = [pavel_mse_y.get(i, np.nan) for i in points]

# Clean data for plots that dislike NaNs (like boxplot)
jara_clean_x = [x for x in jara_vals_x if not np.isnan(x)]
jara_clean_y = [y for y in jara_vals_y if not np.isnan(y)]
pavel_clean_x = [x for x in pavel_vals_x if not np.isnan(x)]
pavel_clean_y = [y for y in pavel_vals_y if not np.isnan(y)]

# Calculate differences, handling NaNs
diff_x = [p - j if not (np.isnan(p) or np.isnan(j)) else np.nan for p, j in zip(pavel_vals_x, jara_vals_x)]
diff_y = [p - j if not (np.isnan(p) or np.isnan(j)) else np.nan for p, j in zip(pavel_vals_y, jara_vals_y)]

# Plot 1: Per-Point MSE Comparison (Line Plot)
fig1, ax1 = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
ax1[0].plot(points, jara_vals_x, marker='o', linestyle='-', label='Jara MSE X')
ax1[0].plot(points, pavel_vals_x, marker='x', linestyle='--', label='Pavel MSE X')
ax1[0].set_ylabel('Average MSE X')
ax1[0].set_title('Per-Point Prediction MSE Comparison')
ax1[0].legend()
ax1[0].grid(True, which='both', linestyle='--', linewidth=0.5)
ax1[0].set_xticks(points)
ax1[1].plot(points, jara_vals_y, marker='o', linestyle='-', label='Jara MSE Y')
ax1[1].plot(points, pavel_vals_y, marker='x', linestyle='--', label='Pavel MSE Y')
ax1[1].set_xlabel('Point Index (0-23)')
ax1[1].set_ylabel('Average MSE Y')
ax1[1].legend()
ax1[1].grid(True, which='both', linestyle='--', linewidth=0.5)
fig1.tight_layout(rect=[0, 0.03, 1, 0.98])
plot1_path = os.path.join(PLOT_DIR, 'per_point_mse_comparison.png')
plt.savefig(plot1_path)
print(f"Saved per-point MSE plot to {plot1_path}")
plt.close(fig1)

# Plot 2: Overall MSE Comparison (Bar Chart)
labels = ['Jara', 'Pavel']
overall_x = [jara_overall_mse_x, pavel_overall_mse_x]
overall_y = [jara_overall_mse_y, pavel_overall_mse_y]
overall_total = [jara_overall_total_mse, pavel_overall_total_mse]

x_pos = np.arange(len(labels))
width = 0.25

fig2, ax2 = plt.subplots(figsize=(10, 6))
rects1 = ax2.bar(x_pos - width, overall_x, width, label='Avg MSE X')
rects2 = ax2.bar(x_pos, overall_y, width, label='Avg MSE Y')
rects3 = ax2.bar(x_pos + width, overall_total, width, label='Avg Total MSE')

ax2.set_ylabel('Overall Average MSE')
ax2.set_title('Overall Prediction MSE Comparison')
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

# Plot 3: Distribution of Per-Point MSEs (Box Plot)
fig3, ax3 = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Box plot for MSE X
if jara_clean_x or pavel_clean_x:
    ax3[0].boxplot([jara_clean_x, pavel_clean_x], labels=['Jara X', 'Pavel X'], showmeans=True)
    ax3[0].set_title('Distribution of Per-Point MSE X')
    ax3[0].set_ylabel('Per-Point MSE')
    ax3[0].grid(True, linestyle='--', linewidth=0.5)
else:
    ax3[0].text(0.5, 0.5, 'No valid X data for boxplot', ha='center', va='center')

# Box plot for MSE Y
if jara_clean_y or pavel_clean_y:
    ax3[1].boxplot([jara_clean_y, pavel_clean_y], labels=['Jara Y', 'Pavel Y'], showmeans=True)
    ax3[1].set_title('Distribution of Per-Point MSE Y')
    ax3[1].grid(True, linestyle='--', linewidth=0.5)
else:
    ax3[1].text(0.5, 0.5, 'No valid Y data for boxplot', ha='center', va='center')

fig3.tight_layout()
plot3_path = os.path.join(PLOT_DIR, 'mse_distribution_comparison.png')
plt.savefig(plot3_path)
print(f"Saved MSE distribution plot to {plot3_path}")
plt.close(fig3)

# Plot 4: Difference in Per-Point MSEs (Pavel - Jara)
fig4, ax4 = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

ax4[0].plot(points, diff_x, marker='.', linestyle='-', label='MSE X Difference (Pavel - Jara)')
ax4[0].axhline(0, color='grey', linestyle='--', linewidth=0.8)
ax4[0].set_ylabel('MSE Difference')
ax4[0].set_title('Difference in Per-Point MSE (Pavel - Jara)')
ax4[0].legend()
ax4[0].grid(True, which='both', linestyle='--', linewidth=0.5)
ax4[0].set_xticks(points)

ax4[1].plot(points, diff_y, marker='.', linestyle='-', label='MSE Y Difference (Pavel - Jara)')
ax4[1].axhline(0, color='grey', linestyle='--', linewidth=0.8)
ax4[1].set_xlabel('Point Index (0-23)')
ax4[1].set_ylabel('MSE Difference')
ax4[1].legend()
ax4[1].grid(True, which='both', linestyle='--', linewidth=0.5)

fig4.tight_layout(rect=[0, 0.03, 1, 0.98])
plot4_path = os.path.join(PLOT_DIR, 'mse_difference_per_point.png')
plt.savefig(plot4_path)
print(f"Saved MSE difference plot to {plot4_path}")
plt.close(fig4)

# plt.show() # Optional