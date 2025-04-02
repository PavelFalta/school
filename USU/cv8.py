path = "data/data-recovery.csv"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import time
import re

# Load data
df = pd.read_csv(path)
print("Original data shape:", df.shape)

# Identify all keypoints in the dataset
all_columns = df.columns
keypoint_pattern = re.compile(r'target_kp(\d+)_[xy]')
keypoint_matches = [keypoint_pattern.match(col) for col in all_columns]
keypoint_ids = sorted(list(set([int(match.group(1)) for match in keypoint_matches if match])))

print(f"Found {len(keypoint_ids)} keypoints: {keypoint_ids}")

# Split data into train and test sets
row_indices = np.arange(len(df))
train_indices, test_indices = train_test_split(row_indices, test_size=0.2, random_state=42)

# Set up results storage
results = []
y_true = pd.DataFrame(index=test_indices)
y_baseline = pd.DataFrame(index=test_indices)
y_rf = pd.DataFrame(index=test_indices)

# Process each keypoint
start_time = time.time()

for kp_id in keypoint_ids:
    print(f"\n{'='*50}")
    print(f"Processing Keypoint {kp_id}")
    print(f"{'='*50}")
    
    kp_prefix = f"kp{kp_id}"
    
    # Extract target columns
    y_col = f"target_{kp_prefix}_y"
    x_col = f"target_{kp_prefix}_x"
    
    # Store ground truth for test set
    y_true[f'{kp_prefix}_y'] = df.loc[test_indices, y_col].values
    y_true[f'{kp_prefix}_x'] = df.loc[test_indices, x_col].values
    
    # Calculate baseline (highest weight) predictions
    weight_cols = [f'pred_{kp_prefix}_val{i}' for i in range(5)]
    highest_weight_idx = np.argmax(df.loc[:, weight_cols].values, axis=1)
    
    baseline_y = np.zeros(len(df))
    baseline_x = np.zeros(len(df))
    
    for i in range(len(df)):
        idx = highest_weight_idx[i]
        baseline_y[i] = df.iloc[i][f'pred_{kp_prefix}_pos{idx}_y']
        baseline_x[i] = df.iloc[i][f'pred_{kp_prefix}_pos{idx}_x']
    
    # Store baseline predictions for test set
    y_baseline[f'{kp_prefix}_y'] = baseline_y[test_indices]
    y_baseline[f'{kp_prefix}_x'] = baseline_x[test_indices]
    
    # Create features for Random Forest
    feature_cols = []
    
    # Position predictions
    for i in range(5):
        feature_cols.extend([f'pred_{kp_prefix}_pos{i}_y', f'pred_{kp_prefix}_pos{i}_x'])
    
    # Weight values
    feature_cols.extend([f'pred_{kp_prefix}_val{i}' for i in range(5)])
    
    # Centroid and sigma
    feature_cols.extend([
        f'pred_{kp_prefix}_centroid_y', f'pred_{kp_prefix}_centroid_x',
        f'pred_{kp_prefix}_sigma_y', f'pred_{kp_prefix}_sigma_x'
    ])
    
    # Extract features and scale
    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split into train and test
    X_train = X_scaled[train_indices]
    X_test = X_scaled[test_indices]
    
    y_train_y = df.loc[train_indices, y_col].values
    y_train_x = df.loc[train_indices, x_col].values
    
    y_test_y = df.loc[test_indices, y_col].values
    y_test_x = df.loc[test_indices, x_col].values
    
    # Train Y-coordinate Random Forest
    print(f"Training Random Forest for {kp_prefix} Y-coordinate...")
    rf_y = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_y.fit(X_train, y_train_y)
    
    # Train X-coordinate Random Forest
    print(f"Training Random Forest for {kp_prefix} X-coordinate...")
    rf_x = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_x.fit(X_train, y_train_x)
    
    # Make predictions
    rf_pred_y = rf_y.predict(X_test)
    rf_pred_x = rf_x.predict(X_test)
    
    # Store Random Forest predictions
    y_rf[f'{kp_prefix}_y'] = rf_pred_y
    y_rf[f'{kp_prefix}_x'] = rf_pred_x
    
    # Calculate MSE for baseline and Random Forest
    baseline_mse = mean_squared_error(
        np.column_stack((y_test_y, y_test_x)),
        np.column_stack((baseline_y[test_indices], baseline_x[test_indices]))
    )
    
    rf_mse = mean_squared_error(
        np.column_stack((y_test_y, y_test_x)),
        np.column_stack((rf_pred_y, rf_pred_x))
    )
    
    # Calculate improvement percentage
    improvement = 100 * (baseline_mse - rf_mse) / baseline_mse
    
    print(f"Baseline MSE: {baseline_mse:.4f}")
    print(f"Random Forest MSE: {rf_mse:.4f}")
    print(f"Improvement: {improvement:.2f}%")
    
    # Store feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance Y': rf_y.feature_importances_,
        'Importance X': rf_x.feature_importances_
    }).sort_values(by='Importance Y', ascending=False)
    
    print("\nTop 5 features for Y-coordinate:")
    print(feature_importance[['Feature', 'Importance Y']].head(5))
    
    print("\nTop 5 features for X-coordinate:")
    print(feature_importance[['Feature', 'Importance X']].sort_values(by='Importance X', ascending=False).head(5))
    
    # Store results
    results.append({
        'kp_id': kp_id,
        'baseline_mse': baseline_mse,
        'rf_mse': rf_mse,
        'improvement': improvement,
        'feature_importance': feature_importance
    })

total_time = time.time() - start_time
print(f"\nTotal processing time: {total_time:.2f} seconds")

# Calculate overall MSE
def calculate_overall_mse(true_df, pred_df):
    true_values = []
    pred_values = []
    
    for kp_id in keypoint_ids:
        kp_prefix = f"kp{kp_id}"
        true_values.append(true_df[[f'{kp_prefix}_y', f'{kp_prefix}_x']].values)
        pred_values.append(pred_df[[f'{kp_prefix}_y', f'{kp_prefix}_x']].values)
    
    true_values = np.concatenate(true_values, axis=1)
    pred_values = np.concatenate(pred_values, axis=1)
    
    return mean_squared_error(true_values, pred_values)

overall_baseline_mse = calculate_overall_mse(y_true, y_baseline)
overall_rf_mse = calculate_overall_mse(y_true, y_rf)
overall_improvement = 100 * (overall_baseline_mse - overall_rf_mse) / overall_baseline_mse

print("\nOverall Results:")
print(f"Baseline MSE: {overall_baseline_mse:.4f}")
print(f"Random Forest MSE: {overall_rf_mse:.4f}")
print(f"Overall improvement: {overall_improvement:.2f}%")

# Create summary DataFrame
results_df = pd.DataFrame([{
    'kp_id': r['kp_id'],
    'baseline_mse': r['baseline_mse'],
    'rf_mse': r['rf_mse'],
    'improvement': r['improvement']
} for r in results])

print("\nResults by keypoint:")
print(results_df)

# Visualize results
plt.figure(figsize=(15, 6))

# Plot MSE comparison by keypoint
plt.subplot(1, 2, 1)
plt.bar(results_df['kp_id'].astype(str), results_df['baseline_mse'], alpha=0.7, label='Baseline (Highest Weight)', color='red')
plt.bar(results_df['kp_id'].astype(str), results_df['rf_mse'], alpha=0.7, label='Random Forest', color='green')
plt.xlabel('Keypoint ID')
plt.ylabel('MSE')
plt.title('MSE by Keypoint')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Plot improvement percentage
plt.subplot(1, 2, 2)
plt.bar(results_df['kp_id'].astype(str), results_df['improvement'], color='blue')
plt.xlabel('Keypoint ID')
plt.ylabel('Improvement (%)')
plt.title('Random Forest Improvement over Baseline')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Visualize sample rows with predictions
plt.figure(figsize=(15, 10))

# Select random sample rows
sample_indices = np.random.choice(test_indices, min(6, len(test_indices)), replace=False)

for i, idx in enumerate(sample_indices, 1):
    plt.subplot(2, 3, i)
    
    # Plot each keypoint
    for kp_id in keypoint_ids:
        kp_prefix = f"kp{kp_id}"
        
        # Get positions
        true_y = y_true.loc[idx, f'{kp_prefix}_y']
        true_x = y_true.loc[idx, f'{kp_prefix}_x']
        
        baseline_y = y_baseline.loc[idx, f'{kp_prefix}_y']
        baseline_x = y_baseline.loc[idx, f'{kp_prefix}_x']
        
        rf_y = y_rf.loc[idx, f'{kp_prefix}_y']
        rf_x = y_rf.loc[idx, f'{kp_prefix}_x']
        
        # Plot positions
        plt.scatter(true_x, true_y, color='blue', marker='o', s=80, label='True' if kp_id == keypoint_ids[0] else "")
        plt.scatter(baseline_x, baseline_y, color='red', marker='s', s=50, label='Baseline' if kp_id == keypoint_ids[0] else "")
        plt.scatter(rf_x, rf_y, color='green', marker='^', s=50, label='Random Forest' if kp_id == keypoint_ids[0] else "")
        
        # Draw lines
        plt.plot([true_x, baseline_x], [true_y, baseline_y], 'r-', alpha=0.3)
        plt.plot([true_x, rf_x], [true_y, rf_y], 'g-', alpha=0.3)
        
        # Annotate keypoint id
        plt.annotate(str(kp_id), (true_x, true_y), fontsize=8, ha='right')
    
    plt.title(f'Row {idx}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    if i == 1:
        plt.legend()
    plt.grid(True)
    plt.axis('equal')

plt.suptitle('True vs Predicted Keypoint Positions', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# Save predictions to CSV
result_df = pd.DataFrame(index=test_indices)

# Add predictions to result dataframe
for kp_id in keypoint_ids:
    kp_prefix = f"kp{kp_id}"
    
    # True values
    result_df[f'true_{kp_prefix}_y'] = y_true[f'{kp_prefix}_y']
    result_df[f'true_{kp_prefix}_x'] = y_true[f'{kp_prefix}_x']
    
    # Baseline predictions
    result_df[f'baseline_{kp_prefix}_y'] = y_baseline[f'{kp_prefix}_y']
    result_df[f'baseline_{kp_prefix}_x'] = y_baseline[f'{kp_prefix}_x']
    
    # Random Forest predictions
    result_df[f'rf_{kp_prefix}_y'] = y_rf[f'{kp_prefix}_y']
    result_df[f'rf_{kp_prefix}_x'] = y_rf[f'{kp_prefix}_x']

# Save to CSV
output_path = "data/keypoint_predictions_simple.csv"
result_df.to_csv(output_path)
print(f"\nPredictions saved to {output_path}")

# Display sample predictions
print("\nSample predictions:")
print(result_df.head())