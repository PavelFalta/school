path = "data/data-recovery.csv"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import time
import re

# Load data
df = pd.read_csv(path)
print("Original data shape:", df.shape)
print(df.head())

# Identify all keypoints in the dataset
all_columns = df.columns
keypoint_pattern = re.compile(r'target_kp(\d+)_[xy]')
keypoint_matches = [keypoint_pattern.match(col) for col in all_columns]
keypoint_ids = sorted(list(set([int(match.group(1)) for match in keypoint_matches if match])))

print(f"Found {len(keypoint_ids)} keypoints: {keypoint_ids}")

# Create data structures to hold results
row_indices = np.arange(len(df))
train_indices, test_indices = train_test_split(row_indices, test_size=0.2, random_state=42)

# Set up results dataframes
y_true = pd.DataFrame(index=test_indices)
y_baseline = pd.DataFrame(index=test_indices)
y_weighted = pd.DataFrame(index=test_indices)
y_ensemble = pd.DataFrame(index=test_indices)

# Function to create engineered features for a keypoint
def create_engineered_features(df, kp_id):
    features = {}
    kp_prefix = f"kp{kp_id}"
    
    # Original features: all positions and weights
    for i in range(5):
        features[f'pos{i}_y'] = df[f'pred_{kp_prefix}_pos{i}_y'].values
        features[f'pos{i}_x'] = df[f'pred_{kp_prefix}_pos{i}_x'].values
        features[f'val{i}'] = df[f'pred_{kp_prefix}_val{i}'].values
    
    # Add centroid and sigma
    features['centroid_y'] = df[f'pred_{kp_prefix}_centroid_y'].values
    features['centroid_x'] = df[f'pred_{kp_prefix}_centroid_x'].values
    features['sigma_y'] = df[f'pred_{kp_prefix}_sigma_y'].values
    features['sigma_x'] = df[f'pred_{kp_prefix}_sigma_x'].values
    
    # Add spatial relationships between predictions
    for i in range(5):
        for j in range(i+1, 5):
            # Distance between predictions
            features[f'dist_{i}_{j}'] = np.sqrt(
                (df[f'pred_{kp_prefix}_pos{i}_y'] - df[f'pred_{kp_prefix}_pos{j}_y'])**2 + 
                (df[f'pred_{kp_prefix}_pos{i}_x'] - df[f'pred_{kp_prefix}_pos{j}_x'])**2
            )
    
    # Add distances from each prediction to centroid
    for i in range(5):
        features[f'dist_to_centroid_{i}'] = np.sqrt(
            (df[f'pred_{kp_prefix}_pos{i}_y'] - df[f'pred_{kp_prefix}_centroid_y'])**2 + 
            (df[f'pred_{kp_prefix}_pos{i}_x'] - df[f'pred_{kp_prefix}_centroid_x'])**2
        )
    
    # Add normalized weights (sum to 1)
    weight_cols = [f'pred_{kp_prefix}_val{i}' for i in range(5)]
    weight_sum = df[weight_cols].sum(axis=1)
    for i in range(5):
        features[f'norm_val{i}'] = df[f'pred_{kp_prefix}_val{i}'].values / weight_sum
    
    # Add weight ratios
    for i in range(5):
        for j in range(i+1, 5):
            # Avoid division by zero
            ratio = df[f'pred_{kp_prefix}_val{i}'] / df[f'pred_{kp_prefix}_val{j}'].replace(0, 1e-10)
            features[f'weight_ratio_{i}_{j}'] = ratio
    
    # Create feature DataFrame
    return pd.DataFrame(features)

# Process each keypoint
results = []
total_start_time = time.time()
models = {}  # Store trained models for future use

# First, train models for each keypoint
for kp_id in keypoint_ids:
    print(f"\n{'='*50}")
    print(f"Training models for Keypoint {kp_id}")
    print(f"{'='*50}")
    
    kp_prefix = f"kp{kp_id}"
    
    # Extract target columns
    y_col = f"target_{kp_prefix}_y"
    x_col = f"target_{kp_prefix}_x"
    
    # Add ground truth to results dataframe
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
    
    # Add baseline predictions to results
    y_baseline[f'{kp_prefix}_y'] = baseline_y[test_indices]
    y_baseline[f'{kp_prefix}_x'] = baseline_x[test_indices]
    
    # Calculate weighted average predictions
    weighted_y = np.zeros(len(df))
    weighted_x = np.zeros(len(df))
    
    for i in range(len(df)):
        weights = df.iloc[i][weight_cols].values
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        for j in range(5):
            weighted_y[i] += weights[j] * df.iloc[i][f'pred_{kp_prefix}_pos{j}_y']
            weighted_x[i] += weights[j] * df.iloc[i][f'pred_{kp_prefix}_pos{j}_x']
    
    # Add weighted predictions to results
    y_weighted[f'{kp_prefix}_y'] = weighted_y[test_indices]
    y_weighted[f'{kp_prefix}_x'] = weighted_x[test_indices]
    
    # Create engineered features
    print(f"Creating engineered features for {kp_prefix}...")
    features_df = create_engineered_features(df, kp_id)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)
    
    # Train models
    X_train = X_scaled[train_indices]
    y_train_y = df.loc[train_indices, y_col].values
    y_train_x = df.loc[train_indices, x_col].values
    
    # Train models for Y coordinate
    ridge_y = Ridge(alpha=0.5)
    svr_y = SVR(kernel='rbf', C=10, epsilon=0.1)
    rf_y = RandomForestRegressor(n_estimators=100, random_state=42)
    
    ridge_y.fit(X_train, y_train_y)
    svr_y.fit(X_train, y_train_y)
    rf_y.fit(X_train, y_train_y)
    
    # Train models for X coordinate
    ridge_x = Ridge(alpha=0.5)
    svr_x = SVR(kernel='rbf', C=10, epsilon=0.1)
    rf_x = RandomForestRegressor(n_estimators=100, random_state=42)
    
    ridge_x.fit(X_train, y_train_x)
    svr_x.fit(X_train, y_train_x)
    rf_x.fit(X_train, y_train_x)
    
    # Store models, scaler and features
    models[kp_id] = {
        'y_models': {
            'ridge': ridge_y,
            'svr': svr_y,
            'rf': rf_y
        },
        'x_models': {
            'ridge': ridge_x,
            'svr': svr_x,
            'rf': rf_x
        },
        'scaler': scaler,
        'features_df': features_df
    }
    
    # Make predictions on test set
    X_test = X_scaled[test_indices]
    
    # Y-coordinate predictions
    ridge_pred_y = ridge_y.predict(X_test)
    svr_pred_y = svr_y.predict(X_test)
    rf_pred_y = rf_y.predict(X_test)
    
    # X-coordinate predictions
    ridge_pred_x = ridge_x.predict(X_test)
    svr_pred_x = svr_x.predict(X_test)
    rf_pred_x = rf_x.predict(X_test)
    
    # Calculate MSEs for each model
    y_test_y = df.loc[test_indices, y_col].values
    y_test_x = df.loc[test_indices, x_col].values
    
    ridge_mse_y = mean_squared_error(y_test_y, ridge_pred_y)
    svr_mse_y = mean_squared_error(y_test_y, svr_pred_y)
    rf_mse_y = mean_squared_error(y_test_y, rf_pred_y)
    
    ridge_mse_x = mean_squared_error(y_test_x, ridge_pred_x)
    svr_mse_x = mean_squared_error(y_test_x, svr_pred_x)
    rf_mse_x = mean_squared_error(y_test_x, rf_pred_x)
    
    # Use inverse error weighting for ensemble
    def inverse_error_weight(error):
        return 1.0 / (error + 1e-10)
    
    # Weight models based on their performance
    y_errors = np.array([ridge_mse_y, svr_mse_y, rf_mse_y])
    y_weights = inverse_error_weight(y_errors)
    y_weights = y_weights / np.sum(y_weights)
    
    x_errors = np.array([ridge_mse_x, svr_mse_x, rf_mse_x])
    x_weights = inverse_error_weight(x_errors)
    x_weights = x_weights / np.sum(x_weights)
    
    # Create ensemble predictions
    ensemble_y = (
        y_weights[0] * ridge_pred_y + 
        y_weights[1] * svr_pred_y + 
        y_weights[2] * rf_pred_y
    )
    
    ensemble_x = (
        x_weights[0] * ridge_pred_x + 
        x_weights[1] * svr_pred_x + 
        x_weights[2] * rf_pred_x
    )
    
    # Add ensemble predictions to results
    y_ensemble[f'{kp_prefix}_y'] = ensemble_y
    y_ensemble[f'{kp_prefix}_x'] = ensemble_x
    
    # Calculate and store metrics
    baseline_mse = mean_squared_error(
        np.column_stack((y_test_y, y_test_x)), 
        np.column_stack((baseline_y[test_indices], baseline_x[test_indices]))
    )
    
    weighted_mse = mean_squared_error(
        np.column_stack((y_test_y, y_test_x)), 
        np.column_stack((weighted_y[test_indices], weighted_x[test_indices]))
    )
    
    ensemble_mse = mean_squared_error(
        np.column_stack((y_test_y, y_test_x)), 
        np.column_stack((ensemble_y, ensemble_x))
    )
    
    # Store results for this keypoint
    keypoint_result = {
        'kp_id': kp_id,
        'baseline_mse': baseline_mse,
        'weighted_mse': weighted_mse,
        'ensemble_mse': ensemble_mse,
        'ridge_mse_y': ridge_mse_y,
        'svr_mse_y': svr_mse_y,
        'rf_mse_y': rf_mse_y,
        'ridge_mse_x': ridge_mse_x,
        'svr_mse_x': svr_mse_x,
        'rf_mse_x': rf_mse_x,
        'y_weights': y_weights,
        'x_weights': x_weights
    }
    
    results.append(keypoint_result)
    print(f"Completed training for {kp_prefix}")

total_time = time.time() - total_start_time
print(f"\nTotal processing time: {total_time:.2f} seconds")

# Create results DataFrame
results_df = pd.DataFrame(results)
print("\nSummary of model performance by keypoint:")
print(results_df[['kp_id', 'baseline_mse', 'weighted_mse', 'ensemble_mse']])

# Calculate overall MSE for each method (across all keypoints)
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
overall_weighted_mse = calculate_overall_mse(y_true, y_weighted)
overall_ensemble_mse = calculate_overall_mse(y_true, y_ensemble)

print("\nOverall MSE (all keypoints):")
print(f"Baseline method: {overall_baseline_mse:.4f}")
print(f"Weighted average method: {overall_weighted_mse:.4f}")
print(f"Ensemble method: {overall_ensemble_mse:.4f}")

# Calculate improvement percentages
baseline_improvement = 100 * (overall_baseline_mse - overall_ensemble_mse) / overall_baseline_mse
weighted_improvement = 100 * (overall_weighted_mse - overall_ensemble_mse) / overall_weighted_mse

print(f"\nEnsemble improvement over baseline: {baseline_improvement:.2f}%")
print(f"Ensemble improvement over weighted average: {weighted_improvement:.2f}%")

# Visualize results
plt.figure(figsize=(12, 6))

# Plot MSE by keypoint
plt.subplot(1, 2, 1)
plt.bar(results_df['kp_id'].astype(str), results_df['baseline_mse'], alpha=0.7, label='Baseline', color='red')
plt.bar(results_df['kp_id'].astype(str), results_df['weighted_mse'], alpha=0.7, label='Weighted Avg', color='green')
plt.bar(results_df['kp_id'].astype(str), results_df['ensemble_mse'], alpha=0.7, label='Ensemble', color='purple')
plt.xlabel('Keypoint ID')
plt.ylabel('MSE')
plt.title('MSE by Keypoint and Method')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Plot overall MSE comparison
plt.subplot(1, 2, 2)
methods = ['Baseline', 'Weighted Avg', 'Ensemble']
overall_mses = [overall_baseline_mse, overall_weighted_mse, overall_ensemble_mse]
bars = plt.bar(methods, overall_mses, color=['red', 'green', 'purple'])
plt.ylabel('MSE')
plt.title('Overall MSE Across All Keypoints')
plt.grid(axis='y', alpha=0.3)

# Highlight the best method
best_idx = np.argmin(overall_mses)
bars[best_idx].set_color('gold')
plt.annotate(f'Best: {overall_mses[best_idx]:.4f}', 
             xy=(best_idx, overall_mses[best_idx]),
             xytext=(best_idx, overall_mses[best_idx] + 1),
             ha='center',
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.show()

# Visualize predicted vs true positions for a sample of test rows
plt.figure(figsize=(15, 10))

# Select a random sample of test rows to visualize
sample_indices = np.random.choice(test_indices, min(5, len(test_indices)), replace=False)

for i, idx in enumerate(sample_indices, 1):
    plt.subplot(2, 3, i)
    
    # Plot each keypoint
    for kp_id in keypoint_ids:
        kp_prefix = f"kp{kp_id}"
        
        # Get true and predicted positions
        true_y = y_true.loc[idx, f'{kp_prefix}_y']
        true_x = y_true.loc[idx, f'{kp_prefix}_x']
        
        # Get ensemble predictions
        ensemble_y = y_ensemble.loc[idx, f'{kp_prefix}_y']
        ensemble_x = y_ensemble.loc[idx, f'{kp_prefix}_x']
        
        # Get weighted average predictions
        weighted_y = y_weighted.loc[idx, f'{kp_prefix}_y']
        weighted_x = y_weighted.loc[idx, f'{kp_prefix}_x']
        
        # Plot true position
        plt.scatter(true_x, true_y, color='blue', s=80, marker='o', 
                   label='True' if kp_id == keypoint_ids[0] else "")
        
        # Plot weighted average prediction
        plt.scatter(weighted_x, weighted_y, color='green', s=50, marker='s',
                   label='Weighted Avg' if kp_id == keypoint_ids[0] else "")
        
        # Plot ensemble prediction
        plt.scatter(ensemble_x, ensemble_y, color='red', s=50, marker='^',
                   label='Ensemble' if kp_id == keypoint_ids[0] else "")
        
        # Draw lines between true and predictions
        plt.plot([true_x, ensemble_x], [true_y, ensemble_y], 'r-', alpha=0.3)
        plt.plot([true_x, weighted_x], [true_y, weighted_y], 'g-', alpha=0.3)
        
        # Annotate keypoint id
        plt.annotate(str(kp_id), (true_x, true_y), fontsize=8, ha='right')
    
    plt.title(f'Row {idx}')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    if i == 1:
        plt.legend()
    plt.grid(True)
    plt.axis('equal')

plt.suptitle('True vs Predicted Keypoint Positions for Sample Rows', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# Save the predictions for all test rows
result_df = pd.DataFrame(index=test_indices)

# Add all true and predicted values to the result dataframe
for kp_id in keypoint_ids:
    kp_prefix = f"kp{kp_id}"
    
    # True values
    result_df[f'true_{kp_prefix}_y'] = y_true[f'{kp_prefix}_y']
    result_df[f'true_{kp_prefix}_x'] = y_true[f'{kp_prefix}_x']
    
    # Baseline predictions
    result_df[f'baseline_{kp_prefix}_y'] = y_baseline[f'{kp_prefix}_y']
    result_df[f'baseline_{kp_prefix}_x'] = y_baseline[f'{kp_prefix}_x']
    
    # Weighted average predictions
    result_df[f'weighted_{kp_prefix}_y'] = y_weighted[f'{kp_prefix}_y']
    result_df[f'weighted_{kp_prefix}_x'] = y_weighted[f'{kp_prefix}_x']
    
    # Ensemble predictions
    result_df[f'ensemble_{kp_prefix}_y'] = y_ensemble[f'{kp_prefix}_y']
    result_df[f'ensemble_{kp_prefix}_x'] = y_ensemble[f'{kp_prefix}_x']

# Save to CSV
output_path = "data/keypoint_predictions.csv"
result_df.to_csv(output_path)
print(f"\nPredictions saved to {output_path}")

# Display a sample of the predictions
print("\nSample of predictions:")
print(result_df.head())

# Function to predict for new data
def predict_keypoints(new_data):
    """
    Predict keypoint positions for new data using trained ensemble models
    
    Args:
        new_data: DataFrame containing the same format as training data
        
    Returns:
        DataFrame with predicted keypoint positions
    """
    predictions = {}
    
    for kp_id in keypoint_ids:
        kp_prefix = f"kp{kp_id}"
        
        # Get the models and scaler for this keypoint
        keypoint_models = models[kp_id]
        
        # Create features
        features = create_engineered_features(new_data, kp_id)
        
        # Scale features
        X_scaled = keypoint_models['scaler'].transform(features)
        
        # Make predictions with each model
        y_models = keypoint_models['y_models']
        x_models = keypoint_models['x_models']
        
        ridge_pred_y = y_models['ridge'].predict(X_scaled)
        svr_pred_y = y_models['svr'].predict(X_scaled)
        rf_pred_y = y_models['rf'].predict(X_scaled)
        
        ridge_pred_x = x_models['ridge'].predict(X_scaled)
        svr_pred_x = x_models['svr'].predict(X_scaled)
        rf_pred_x = x_models['rf'].predict(X_scaled)
        
        # Get weights from results
        kp_result = results_df[results_df['kp_id'] == kp_id].iloc[0]
        y_weights = kp_result['y_weights']
        x_weights = kp_result['x_weights']
        
        # Create ensemble predictions
        ensemble_y = (
            y_weights[0] * ridge_pred_y + 
            y_weights[1] * svr_pred_y + 
            y_weights[2] * rf_pred_y
        )
        
        ensemble_x = (
            x_weights[0] * ridge_pred_x + 
            x_weights[1] * svr_pred_x + 
            x_weights[2] * rf_pred_x
        )
        
        # Store predictions
        predictions[f'{kp_prefix}_y'] = ensemble_y
        predictions[f'{kp_prefix}_x'] = ensemble_x
    
    return pd.DataFrame(predictions)