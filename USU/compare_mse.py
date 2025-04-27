import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_jara = "data/predikce_jara"
path_pavel = "data/predikce_pavel"

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
jara_overall_mse_x = np.nanmean(list(jara_mse_x.values()))
jara_overall_mse_y = np.nanmean(list(jara_mse_y.values()))
jara_overall_total_mse = jara_overall_mse_x + jara_overall_mse_y

pavel_overall_mse_x = np.nanmean(list(pavel_mse_x.values()))
pavel_overall_mse_y = np.nanmean(list(pavel_mse_y.values()))
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
jara_vals_x = [jara_mse_x.get(i, np.nan) for i in points]
jara_vals_y = [jara_mse_y.get(i, np.nan) for i in points]
pavel_vals_x = [pavel_mse_x.get(i, np.nan) for i in points]
pavel_vals_y = [pavel_mse_y.get(i, np.nan) for i in points]


fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot X MSE
ax[0].plot(points, jara_vals_x, marker='o', linestyle='-', label='Jara MSE X')
ax[0].plot(points, pavel_vals_x, marker='x', linestyle='--', label='Pavel MSE X')
ax[0].set_ylabel('Average MSE X')
ax[0].set_title('Per-Point Prediction MSE Comparison')
ax[0].legend()
ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
ax[0].set_xticks(points) # Ensure all points are marked

# Plot Y MSE
ax[1].plot(points, jara_vals_y, marker='o', linestyle='-', label='Jara MSE Y')
ax[1].plot(points, pavel_vals_y, marker='x', linestyle='--', label='Pavel MSE Y')
ax[1].set_xlabel('Point Index (0-23)')
ax[1].set_ylabel('Average MSE Y')
ax[1].legend()
ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)


fig.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout slightly
plt.savefig('per_point_mse_comparison.png')
print("Saved per-point MSE comparison plot to per_point_mse_comparison.png")

# plt.show() # Optional