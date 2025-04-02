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
y_centroid = pd.DataFrame(index=test_indices)
y_rf = pd.DataFrame(index=test_indices)
y_rf_centroid = pd.DataFrame(index=test_indices)

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
    
    # Store centroid values for comparison
    y_centroid[f'{kp_prefix}_y'] = df.loc[test_indices, f'pred_{kp_prefix}_centroid_y'].values
    y_centroid[f'{kp_prefix}_x'] = df.loc[test_indices, f'pred_{kp_prefix}_centroid_x'].values
    
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
    
    # Create two sets of features:
    # 1. Full feature set
    # 2. Centroid-only feature set
    
    # 1. Full feature set
    full_feature_cols = []
    
    # Position predictions
    for i in range(5):
        full_feature_cols.extend([f'pred_{kp_prefix}_pos{i}_y', f'pred_{kp_prefix}_pos{i}_x'])
    
    # Weight values
    full_feature_cols.extend([f'pred_{kp_prefix}_val{i}' for i in range(5)])
    
    # Centroid and sigma
    full_feature_cols.extend([
        f'pred_{kp_prefix}_centroid_y', f'pred_{kp_prefix}_centroid_x',
        f'pred_{kp_prefix}_sigma_y', f'pred_{kp_prefix}_sigma_x'
    ])
    
    # 2. Centroid-only feature set
    centroid_feature_cols = [
        f'pred_{kp_prefix}_centroid_y', f'pred_{kp_prefix}_centroid_x',
        f'pred_{kp_prefix}_sigma_y', f'pred_{kp_prefix}_sigma_x'
    ]
    
    # Extract and scale full features
    X_full = df[full_feature_cols].values
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full)
    
    # Extract and scale centroid features
    X_centroid = df[centroid_feature_cols].values
    scaler_centroid = StandardScaler()
    X_centroid_scaled = scaler_centroid.fit_transform(X_centroid)
    
    # Split into train and test
    X_full_train = X_full_scaled[train_indices]
    X_full_test = X_full_scaled[test_indices]
    
    X_centroid_train = X_centroid_scaled[train_indices]
    X_centroid_test = X_centroid_scaled[test_indices]
    
    y_train_y = df.loc[train_indices, y_col].values
    y_train_x = df.loc[train_indices, x_col].values
    
    y_test_y = df.loc[test_indices, y_col].values
    y_test_x = df.loc[test_indices, x_col].values
    
    # Train Y-coordinate Random Forest (full features)
    print(f"Training RF with full features for {kp_prefix} Y-coordinate...")
    rf_y = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_y.fit(X_full_train, y_train_y)
    
    # Train X-coordinate Random Forest (full features)
    print(f"Training RF with full features for {kp_prefix} X-coordinate...")
    rf_x = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_x.fit(X_full_train, y_train_x)
    
    # Train Y-coordinate Random Forest (centroid-only features)
    print(f"Training RF with centroids for {kp_prefix} Y-coordinate...")
    rf_centroid_y = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_centroid_y.fit(X_centroid_train, y_train_y)
    
    # Train X-coordinate Random Forest (centroid-only features)
    print(f"Training RF with centroids for {kp_prefix} X-coordinate...")
    rf_centroid_x = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_centroid_x.fit(X_centroid_train, y_train_x)
    
    # Make predictions (full features)
    rf_pred_y = rf_y.predict(X_full_test)
    rf_pred_x = rf_x.predict(X_full_test)
    
    # Make predictions (centroid-only features)
    rf_centroid_pred_y = rf_centroid_y.predict(X_centroid_test)
    rf_centroid_pred_x = rf_centroid_x.predict(X_centroid_test)
    
    # Store Random Forest predictions
    y_rf[f'{kp_prefix}_y'] = rf_pred_y
    y_rf[f'{kp_prefix}_x'] = rf_pred_x
    
    # Store Random Forest centroid-only predictions
    y_rf_centroid[f'{kp_prefix}_y'] = rf_centroid_pred_y
    y_rf_centroid[f'{kp_prefix}_x'] = rf_centroid_pred_x
    
    # Calculate MSE for all methods
    baseline_mse = mean_squared_error(
        np.column_stack((y_test_y, y_test_x)),
        np.column_stack((baseline_y[test_indices], baseline_x[test_indices]))
    )
    
    centroid_mse = mean_squared_error(
        np.column_stack((y_test_y, y_test_x)),
        np.column_stack((df.loc[test_indices, f'pred_{kp_prefix}_centroid_y'], 
                          df.loc[test_indices, f'pred_{kp_prefix}_centroid_x']))
    )
    
    rf_mse = mean_squared_error(
        np.column_stack((y_test_y, y_test_x)),
        np.column_stack((rf_pred_y, rf_pred_x))
    )
    
    rf_centroid_mse = mean_squared_error(
        np.column_stack((y_test_y, y_test_x)),
        np.column_stack((rf_centroid_pred_y, rf_centroid_pred_x))
    )
    
    # Calculate improvement percentages
    baseline_to_rf_improvement = 100 * (baseline_mse - rf_mse) / baseline_mse
    centroid_to_rf_centroid_improvement = 100 * (centroid_mse - rf_centroid_mse) / centroid_mse
    centroid_to_baseline_improvement = 100 * (baseline_mse - centroid_mse) / baseline_mse
    
    print(f"Baseline MSE: {baseline_mse:.4f}")
    print(f"Centroid MSE: {centroid_mse:.4f}")
    print(f"RF (full) MSE: {rf_mse:.4f}")
    print(f"RF (centroid) MSE: {rf_centroid_mse:.4f}")
    
    print(f"Improvement (Baseline → RF full): {baseline_to_rf_improvement:.2f}%")
    print(f"Improvement (Centroid → RF centroid): {centroid_to_rf_centroid_improvement:.2f}%")
    print(f"Improvement (Baseline → Centroid): {centroid_to_baseline_improvement:.2f}%")
    
    # Store feature importance for full model
    feature_importance = pd.DataFrame({
        'Feature': full_feature_cols,
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
        'centroid_mse': centroid_mse,
        'rf_mse': rf_mse,
        'rf_centroid_mse': rf_centroid_mse,
        'baseline_to_rf_improvement': baseline_to_rf_improvement,
        'centroid_to_rf_centroid_improvement': centroid_to_rf_centroid_improvement,
        'centroid_to_baseline_improvement': centroid_to_baseline_improvement,
        'feature_importance': feature_importance
    })

total_time = time.time() - start_time
print(f"\nTotal processing time: {total_time:.2f} seconds")

# Calculate overall MSE for each method
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
overall_centroid_mse = calculate_overall_mse(y_true, y_centroid)
overall_rf_mse = calculate_overall_mse(y_true, y_rf)
overall_rf_centroid_mse = calculate_overall_mse(y_true, y_rf_centroid)

overall_baseline_to_rf = 100 * (overall_baseline_mse - overall_rf_mse) / overall_baseline_mse
overall_centroid_to_rf_centroid = 100 * (overall_centroid_mse - overall_rf_centroid_mse) / overall_centroid_mse
overall_baseline_to_centroid = 100 * (overall_baseline_mse - overall_centroid_mse) / overall_baseline_mse

print("\nOverall Results:")
print(f"Baseline MSE: {overall_baseline_mse:.4f}")
print(f"Centroid MSE: {overall_centroid_mse:.4f}")
print(f"RF (full) MSE: {overall_rf_mse:.4f}")
print(f"RF (centroid) MSE: {overall_rf_centroid_mse:.4f}")

print(f"\nImprovement (Baseline → RF full): {overall_baseline_to_rf:.2f}%")
print(f"Improvement (Centroid → RF centroid): {overall_centroid_to_rf_centroid:.2f}%")
print(f"Improvement (Baseline → Centroid): {overall_baseline_to_centroid:.2f}%")

# Create summary DataFrame
results_df = pd.DataFrame([{
    'kp_id': r['kp_id'],
    'baseline_mse': r['baseline_mse'],
    'centroid_mse': r['centroid_mse'],
    'rf_mse': r['rf_mse'],
    'rf_centroid_mse': r['rf_centroid_mse'],
    'baseline_to_rf_improvement': r['baseline_to_rf_improvement'],
    'centroid_to_rf_centroid_improvement': r['centroid_to_rf_centroid_improvement'],
    'centroid_to_baseline_improvement': r['centroid_to_baseline_improvement']
} for r in results])

print("\nResults by keypoint:")
print(results_df)

# Visualize results
plt.figure(figsize=(16, 8))

# Plot MSE comparison by keypoint
plt.subplot(1, 2, 1)
bar_width = 0.2
index = np.arange(len(keypoint_ids))

plt.bar(index - bar_width*1.5, results_df['baseline_mse'], bar_width, label='Baseline', color='red', alpha=0.7)
plt.bar(index - bar_width*0.5, results_df['centroid_mse'], bar_width, label='Centroid', color='blue', alpha=0.7)
plt.bar(index + bar_width*0.5, results_df['rf_mse'], bar_width, label='RF (full)', color='green', alpha=0.7)
plt.bar(index + bar_width*1.5, results_df['rf_centroid_mse'], bar_width, label='RF (centroid)', color='purple', alpha=0.7)

plt.xlabel('Keypoint ID')
plt.ylabel('MSE')
plt.title('MSE by Keypoint and Method')
plt.xticks(index, results_df['kp_id'].astype(str))
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Plot overall MSE comparison
plt.subplot(1, 2, 2)
methods = ['Baseline', 'Centroid', 'RF (full)', 'RF (centroid)']
overall_mses = [overall_baseline_mse, overall_centroid_mse, overall_rf_mse, overall_rf_centroid_mse]
colors = ['red', 'blue', 'green', 'purple']


bars = plt.bar(methods, overall_mses, color=colors, alpha=0.7)
plt.ylabel('MSE')
plt.title('Overall MSE Across All Keypoints')
plt.grid(axis='y', alpha=0.3)

# Highlight the best method
best_idx = np.argmin(overall_mses)
bars[best_idx].set_alpha(1.0)
plt.annotate(f'Best: {overall_mses[best_idx]:.4f}', 
            xy=(best_idx, overall_mses[best_idx]),
            xytext=(best_idx, overall_mses[best_idx] + 1),
            ha='center',
            arrowprops=dict(facecolor='black', shrink=0.05))

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
        
        centroid_y = y_centroid.loc[idx, f'{kp_prefix}_y']
        centroid_x = y_centroid.loc[idx, f'{kp_prefix}_x']
        
        rf_centroid_y = y_rf_centroid.loc[idx, f'{kp_prefix}_y']
        rf_centroid_x = y_rf_centroid.loc[idx, f'{kp_prefix}_x']
        
        # Plot positions
        plt.scatter(true_x, true_y, color='blue', marker='o', s=80, label='True' if kp_id == keypoint_ids[0] else "")
        plt.scatter(baseline_x, baseline_y, color='red', marker='s', s=50, label='Baseline' if kp_id == keypoint_ids[0] else "")
        plt.scatter(centroid_x, centroid_y, color='cyan', marker='d', s=50, label='Centroid' if kp_id == keypoint_ids[0] else "")
        plt.scatter(rf_centroid_x, rf_centroid_y, color='purple', marker='^', s=50, label='RF (centroid)' if kp_id == keypoint_ids[0] else "")
        
        # Draw lines between true and predictions
        plt.plot([true_x, baseline_x], [true_y, baseline_y], 'r-', alpha=0.2)
        plt.plot([true_x, centroid_x], [true_y, centroid_y], 'c-', alpha=0.2)
        plt.plot([true_x, rf_centroid_x], [true_y, rf_centroid_y], 'purple', alpha=0.2)
        
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
    
    # Centroid predictions
    result_df[f'centroid_{kp_prefix}_y'] = y_centroid[f'{kp_prefix}_y']
    result_df[f'centroid_{kp_prefix}_x'] = y_centroid[f'{kp_prefix}_x']
    
    # Random Forest (centroid) predictions
    result_df[f'rf_centroid_{kp_prefix}_y'] = y_rf_centroid[f'{kp_prefix}_y']
    result_df[f'rf_centroid_{kp_prefix}_x'] = y_rf_centroid[f'{kp_prefix}_x']

# Save to CSV
output_path = "data/keypoint_predictions_centroid.csv"
result_df.to_csv(output_path)
print(f"\nPredictions saved to {output_path}")

# Display sample predictions
print("\nSample predictions:")
print(result_df.head())