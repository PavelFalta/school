import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path_jara = "data/predikce_jara"
path_pavel = "data/predikce_pavel"

# csv colums: skutecna_kp0_y,skutecna_kp0_x,nejvetsi_vaha_kp0_y,nejvetsi_vaha_kp0_x,predikce_kp0_y,predikce_kp0_x
# from 0 to 23
# udelat MSE v X, MSE v Y a soucet MSE X+Y

def calculate_mse(df):
    """Calculates MSE for predictions and baselines against the target."""
    mse_pred_x = np.mean((df['skutecna_kp0_x'] - df['predikce_kp0_x'])**2)
    mse_pred_y = np.mean((df['skutecna_kp0_y'] - df['predikce_kp0_y'])**2)
    mse_pred_total = mse_pred_x + mse_pred_y

    mse_base_x = np.mean((df['skutecna_kp0_x'] - df['nejvetsi_vaha_kp0_x'])**2)
    mse_base_y = np.mean((df['skutecna_kp0_y'] - df['nejvetsi_vaha_kp0_y'])**2)
    mse_base_total = mse_base_x + mse_base_y

    return {
        'pred_x': mse_pred_x, 'pred_y': mse_pred_y, 'pred_total': mse_pred_total,
        'base_x': mse_base_x, 'base_y': mse_base_y, 'base_total': mse_base_total
    }

def process_directory(dir_path):
    """Processes all CSV files in a directory and calculates average MSEs."""
    all_mse = {'pred_x': [], 'pred_y': [], 'pred_total': [],
               'base_x': [], 'base_y': [], 'base_total': []}
    
    if not os.path.isdir(dir_path):
        print(f"Error: Directory not found - {dir_path}")
        return None # Return None if directory doesn't exist

    for filename in os.listdir(dir_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(dir_path, filename)
            try:
                # Skip header row if it exists, otherwise assume no header
                df = pd.read_csv(file_path) # Assume header exists based on column names provided
                
                # Basic validation: Check if required columns exist
                required_cols = ['skutecna_kp0_y', 'skutecna_kp0_x', 
                                 'nejvetsi_vaha_kp0_y', 'nejvetsi_vaha_kp0_x', 
                                 'predikce_kp0_y', 'predikce_kp0_x']
                if not all(col in df.columns for col in required_cols):
                    print(f"Warning: Skipping file {filename} due to missing columns.")
                    continue

                # Convert relevant columns to numeric, coercing errors
                for col in required_cols:
                     df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Drop rows with NaN values that might result from coercion
                df.dropna(subset=required_cols, inplace=True)

                if df.empty:
                    print(f"Warning: Skipping file {filename} due to no valid numeric data after cleaning.")
                    continue

                mse_results = calculate_mse(df)
                for key in all_mse:
                    all_mse[key].append(mse_results[key])
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    if not any(all_mse.values()): # Check if any lists are non-empty
         print(f"Warning: No valid data processed in directory {dir_path}")
         # Return dictionary with NaNs or handle as appropriate
         return {key: np.nan for key in all_mse} 


    # Calculate average MSEs across all files
    avg_mse = {key: np.mean(values) for key, values in all_mse.items() if values}
    return avg_mse

# --- Main Execution ---
print("Processing Jara's data...")
mse_jara = process_directory(path_jara)

print("Processing Pavel's data...")
mse_pavel = process_directory(path_pavel)

# Check if processing was successful for both
if mse_jara is None or mse_pavel is None or pd.isna(list(mse_jara.values())).any() or pd.isna(list(mse_pavel.values())).any():
    print("Error: Could not calculate MSE for one or both researchers. Exiting.")
    exit()


# --- Comparison ---
print("--- MSE Results (Average across files) ---")
print(f"Jara - Prediction Total MSE: {mse_jara['pred_total']:.4f}")
print(f"Pavel - Prediction Total MSE: {mse_pavel['pred_total']:.4f}")
print(f"Jara - Baseline Total MSE: {mse_jara['base_total']:.4f}")
print(f"Pavel - Baseline Total MSE: {mse_pavel['base_total']:.4f}")


if mse_jara['pred_total'] < mse_pavel['pred_total']:
    print("Jara's model has a lower average prediction MSE.")
elif mse_pavel['pred_total'] < mse_jara['pred_total']:
    print("Pavel's model has a lower average prediction MSE.")
else:
    print("Both models have the same average prediction MSE.")

# --- Visualization ---
labels = ['Jara', 'Pavel']
pred_mse_x = [mse_jara['pred_x'], mse_pavel['pred_x']]
pred_mse_y = [mse_jara['pred_y'], mse_pavel['pred_y']]
pred_mse_total = [mse_jara['pred_total'], mse_pavel['pred_total']]

base_mse_x = [mse_jara['base_x'], mse_pavel['base_x']]
base_mse_y = [mse_jara['base_y'], mse_pavel['base_y']]
base_mse_total = [mse_jara['base_total'], mse_pavel['base_total']]

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

# Prediction MSE Plot
fig1, ax1 = plt.subplots(figsize=(10, 6))
rects1 = ax1.bar(x - width, pred_mse_x, width, label='MSE X')
rects2 = ax1.bar(x, pred_mse_y, width, label='MSE Y')
rects3 = ax1.bar(x + width, pred_mse_total, width, label='MSE Total (X+Y)')

ax1.set_ylabel('Average MSE')
ax1.set_title('Prediction MSE Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend()
ax1.bar_label(rects1, padding=3, fmt='%.2f')
ax1.bar_label(rects2, padding=3, fmt='%.2f')
ax1.bar_label(rects3, padding=3, fmt='%.2f')
fig1.tight_layout()
plt.savefig('prediction_mse_comparison.png') # Save the plot
print("Saved prediction MSE comparison plot to prediction_mse_comparison.png")


# Baseline MSE Plot
fig2, ax2 = plt.subplots(figsize=(10, 6))
rects4 = ax2.bar(x - width, base_mse_x, width, label='MSE X')
rects5 = ax2.bar(x, base_mse_y, width, label='MSE Y')
rects6 = ax2.bar(x + width, base_mse_total, width, label='MSE Total (X+Y)')

ax2.set_ylabel('Average MSE')
ax2.set_title('Baseline MSE Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.legend()
ax2.bar_label(rects4, padding=3, fmt='%.2f')
ax2.bar_label(rects5, padding=3, fmt='%.2f')
ax2.bar_label(rects6, padding=3, fmt='%.2f')
fig2.tight_layout()
plt.savefig('baseline_mse_comparison.png') # Save the plot
print("Saved baseline MSE comparison plot to baseline_mse_comparison.png")

# plt.show() # Optional: Show plots if running interactively