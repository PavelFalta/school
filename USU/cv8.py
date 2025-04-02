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
from sklearn.neural_network import MLPRegressor
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

# Function to calculate baseline predictions (highest weight approach)
def calculate_baseline_pred(df, kp_id):
    kp_prefix = f"kp{kp_id}"
    weight_cols = [f'pred_{kp_prefix}_val{i}' for i in range(5)]
    prediction_weights = df[weight_cols]
    highest_weight_idx = np.argmax(prediction_weights.values, axis=1)
    
    # Extract the corresponding coordinates based on highest weight
    baseline_pred_y = np.zeros(len(df))
    baseline_pred_x = np.zeros(len(df))
    
    for i in range(len(df)):
        idx = highest_weight_idx[i]
        baseline_pred_y[i] = df.iloc[i][f'pred_{kp_prefix}_pos{idx}_y']
        baseline_pred_x[i] = df.iloc[i][f'pred_{kp_prefix}_pos{idx}_x']
    
    return np.column_stack((baseline_pred_y, baseline_pred_x))

# Function to calculate weighted average predictions
def calculate_weighted_avg_pred(df, kp_id):
    kp_prefix = f"kp{kp_id}"
    weight_cols = [f'pred_{kp_prefix}_val{i}' for i in range(5)]
    prediction_weights = df[weight_cols]
    
    weighted_pred_y = np.zeros(len(df))
    weighted_pred_x = np.zeros(len(df))
    
    for i in range(len(df)):
        weights = prediction_weights.iloc[i].values
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights)
        
        for j in range(5):
            weighted_pred_y[i] += weights[j] * df.iloc[i][f'pred_{kp_prefix}_pos{j}_y']
            weighted_pred_x[i] += weights[j] * df.iloc[i][f'pred_{kp_prefix}_pos{j}_x']
    
    return np.column_stack((weighted_pred_y, weighted_pred_x))

# Function to train ensemble models for a keypoint
def train_ensemble_models(X_train, Y_train, X_test, y_col, x_col):
    # Ridge regression model
    ridge_y = Ridge(alpha=0.5)
    ridge_x = Ridge(alpha=0.5)
    
    # SVR model
    svr_y = SVR(kernel='rbf', C=10, epsilon=0.1)
    svr_x = SVR(kernel='rbf', C=10, epsilon=0.1)
    
    # Random Forest model
    rf_y = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_x = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train each model on Y coordinate
    ridge_y.fit(X_train, Y_train[y_col])
    svr_y.fit(X_train, Y_train[y_col])
    rf_y.fit(X_train, Y_train[y_col])
    
    # Train each model on X coordinate
    ridge_x.fit(X_train, Y_train[x_col])
    svr_x.fit(X_train, Y_train[x_col])
    rf_x.fit(X_train, Y_train[x_col])
    
    # Make predictions
    ridge_pred_y = ridge_y.predict(X_test)
    svr_pred_y = svr_y.predict(X_test)
    rf_pred_y = rf_y.predict(X_test)
    
    ridge_pred_x = ridge_x.predict(X_test)
    svr_pred_x = svr_x.predict(X_test)
    rf_pred_x = rf_x.predict(X_test)
    
    return {
        'ridge': (ridge_pred_y, ridge_pred_x),
        'svr': (svr_pred_y, svr_pred_x),
        'rf': (rf_pred_y, rf_pred_x)
    }

# Function for inverse error weighting
def inverse_error_weight(error):
    return 1.0 / (error + 1e-10)  # Adding small constant to avoid division by zero

# Function to create ensemble prediction
def create_ensemble_prediction(predictions, Y_test, y_col, x_col):
    # Calculate individual model MSEs for Y coordinate
    ridge_mse_y = mean_squared_error(Y_test[y_col], predictions['ridge'][0])
    svr_mse_y = mean_squared_error(Y_test[y_col], predictions['svr'][0])
    rf_mse_y = mean_squared_error(Y_test[y_col], predictions['rf'][0])
    
    # Calculate individual model MSEs for X coordinate
    ridge_mse_x = mean_squared_error(Y_test[x_col], predictions['ridge'][1])
    svr_mse_x = mean_squared_error(Y_test[x_col], predictions['svr'][1])
    rf_mse_x = mean_squared_error(Y_test[x_col], predictions['rf'][1])
    
    # Calculate weights for Y coordinate models based on inverse error
    y_errors = np.array([ridge_mse_y, svr_mse_y, rf_mse_y])
    y_weights = inverse_error_weight(y_errors)
    y_weights = y_weights / np.sum(y_weights)  # Normalize to sum to 1
    
    # Calculate weights for X coordinate models
    x_errors = np.array([ridge_mse_x, svr_mse_x, rf_mse_x])
    x_weights = inverse_error_weight(x_errors)
    x_weights = x_weights / np.sum(x_weights)  # Normalize to sum to 1
    
    # Create weighted ensemble predictions
    ensemble_pred_y = (
        y_weights[0] * predictions['ridge'][0] + 
        y_weights[1] * predictions['svr'][0] + 
        y_weights[2] * predictions['rf'][0]
    )
    
    ensemble_pred_x = (
        x_weights[0] * predictions['ridge'][1] + 
        x_weights[1] * predictions['svr'][1] + 
        x_weights[2] * predictions['rf'][1]
    )
    
    # Compute overall model MSEs
    ridge_mse = 0.5 * ridge_mse_y + 0.5 * ridge_mse_x
    svr_mse = 0.5 * svr_mse_y + 0.5 * svr_mse_x
    rf_mse = 0.5 * rf_mse_y + 0.5 * rf_mse_x
    
    model_mses = {
        'Ridge': ridge_mse,
        'SVR': svr_mse,
        'RF': rf_mse
    }
    
    model_weights = {
        'Y weights': y_weights,
        'X weights': x_weights
    }
    
    return np.column_stack((ensemble_pred_y, ensemble_pred_x)), model_mses, model_weights

# Main function to process a single keypoint
def process_keypoint(df, kp_id, plot=False):
    start_time = time.time()
    print(f"\n{'='*50}")
    print(f"Processing Keypoint {kp_id}")
    print(f"{'='*50}")
    
    kp_prefix = f"kp{kp_id}"
    
    # Extract columns related to this keypoint
    kp_cols = [col for col in df.columns if kp_prefix in col]
    kp_data = df[kp_cols]
    
    # Define target columns
    y_col = f"target_{kp_prefix}_y"
    x_col = f"target_{kp_prefix}_x"
    Y = kp_data[[y_col, x_col]]
    
    # Calculate baseline predictions (highest weight)
    baseline_pred = calculate_baseline_pred(kp_data, kp_id)
    baseline_mse = mean_squared_error(Y, baseline_pred)
    print(f"Baseline MSE (highest weight): {baseline_mse:.4f}")
    
    # Calculate weighted average predictions
    weighted_avg_pred = calculate_weighted_avg_pred(kp_data, kp_id)
    weighted_avg_mse = mean_squared_error(Y, weighted_avg_pred)
    print(f"Weighted Average MSE: {weighted_avg_mse:.4f}")
    
    # Create engineered features
    engineered_features = create_engineered_features(kp_data, kp_id)
    print(f"Engineered features shape: {engineered_features.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(engineered_features)
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
    
    # Get indices for test set
    test_indices = np.array(range(len(Y)))[X_train.shape[0]:]
    baseline_test_pred = baseline_pred[test_indices]
    weighted_test_pred = weighted_avg_pred[test_indices]
    
    # Train ensemble models
    predictions = train_ensemble_models(X_train, Y_train, X_test, y_col, x_col)
    
    # Create ensemble prediction
    ensemble_pred, model_mses, model_weights = create_ensemble_prediction(
        predictions, Y_test, y_col, x_col
    )
    ensemble_mse = mean_squared_error(Y_test, ensemble_pred)
    
    # Print model MSEs
    print("\nIndividual Model MSEs:")
    for model, mse in model_mses.items():
        print(f"{model}: {mse:.4f}")
    
    print(f"\nEnsemble Model MSE: {ensemble_mse:.4f}")
    print(f"Model weights: {model_weights}")
    
    # Calculate test MSEs for all methods
    methods = [
        "Baseline (Highest Weight)", 
        "Weighted Average",
        "Ensemble"
    ]
    
    mse_values = [
        mean_squared_error(Y_test, baseline_test_pred),
        mean_squared_error(Y_test, weighted_test_pred),
        ensemble_mse
    ]
    
    # Find best method
    best_method_idx = np.argmin(mse_values)
    best_method = methods[best_method_idx]
    best_mse = mse_values[best_method_idx]
    
    print(f"\nBest method: {best_method} with MSE: {best_mse:.4f}")
    
    processing_time = time.time() - start_time
    print(f"Processing time: {processing_time:.2f} seconds")
    
    # Optional plotting
    if plot:
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Ground truth vs baseline
        plt.subplot(1, 3, 1)
        plt.scatter(Y_test[x_col], Y_test[y_col], 
                   alpha=0.5, label='Ground truth', color='blue')
        plt.scatter(baseline_test_pred[:, 1], baseline_test_pred[:, 0], 
                   alpha=0.5, label='Highest weight', color='red')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title(f'Ground Truth vs Highest Weight\nMSE: {mse_values[0]:.4f}')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # Plot 2: Ground truth vs weighted average
        plt.subplot(1, 3, 2)
        plt.scatter(Y_test[x_col], Y_test[y_col], 
                   alpha=0.5, label='Ground truth', color='blue')
        plt.scatter(weighted_test_pred[:, 1], weighted_test_pred[:, 0], 
                   alpha=0.5, label='Weighted average', color='green')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title(f'Ground Truth vs Weighted Average\nMSE: {mse_values[1]:.4f}')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # Plot 3: Ground truth vs ensemble
        plt.subplot(1, 3, 3)
        plt.scatter(Y_test[x_col], Y_test[y_col], 
                   alpha=0.5, label='Ground truth', color='blue')
        plt.scatter(ensemble_pred[:, 1], ensemble_pred[:, 0], 
                   alpha=0.5, label='Ensemble', color='purple')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title(f'Ground Truth vs Ensemble\nMSE: {mse_values[2]:.4f}')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        plt.suptitle(f'Keypoint {kp_id} Predictions')
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()
        
        # Plot comparison of MSE values
        plt.figure(figsize=(8, 5))
        bars = plt.bar(methods, mse_values, color=['red', 'green', 'purple'])
        plt.ylabel('Mean Squared Error (MSE)')
        plt.title(f'Comparison of Methods for Keypoint {kp_id}')
        plt.xticks(rotation=45, ha='right')
        
        # Highlight the best method
        bars[best_method_idx].set_color('gold')
        plt.annotate(f'Best: {best_mse:.4f}', 
                     xy=(best_method_idx, best_mse),
                     xytext=(best_method_idx, best_mse + 1),
                     ha='center',
                     arrowprops=dict(facecolor='black', shrink=0.05))
        
        plt.tight_layout()
        plt.show()
    
    return {
        'kp_id': kp_id,
        'baseline_mse': baseline_mse,
        'weighted_avg_mse': weighted_avg_mse,
        'ensemble_mse': ensemble_mse,
        'best_method': best_method,
        'best_mse': best_mse,
        'model_mses': model_mses,
        'processing_time': processing_time
    }

# Process all keypoints
results = []
total_start_time = time.time()

for kp_id in keypoint_ids:
    result = process_keypoint(df, kp_id, plot=(kp_id == keypoint_ids[0]))  # Only plot the first keypoint
    results.append(result)

total_time = time.time() - total_start_time
print(f"\nTotal processing time: {total_time:.2f} seconds")

# Summarize results
print("\n" + "="*80)
print("SUMMARY OF RESULTS FOR ALL KEYPOINTS")
print("="*80)

results_df = pd.DataFrame(results)
print(results_df[['kp_id', 'baseline_mse', 'weighted_avg_mse', 'ensemble_mse', 'best_method', 'best_mse']])

# Calculate improvement percentages
results_df['baseline_to_ensemble_improvement'] = 100 * (results_df['baseline_mse'] - results_df['ensemble_mse']) / results_df['baseline_mse']
results_df['weighted_to_ensemble_improvement'] = 100 * (results_df['weighted_avg_mse'] - results_df['ensemble_mse']) / results_df['weighted_avg_mse']
print("\nImprovement percentages:")
print(results_df[['kp_id', 'baseline_to_ensemble_improvement', 'weighted_to_ensemble_improvement']])

# Plot overall comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(results_df['kp_id'].astype(str), results_df['baseline_mse'], color='red', alpha=0.7, label='Baseline')
plt.bar(results_df['kp_id'].astype(str), results_df['weighted_avg_mse'], color='green', alpha=0.7, label='Weighted Avg')
plt.bar(results_df['kp_id'].astype(str), results_df['ensemble_mse'], color='purple', alpha=0.7, label='Ensemble')
plt.xlabel('Keypoint ID')
plt.ylabel('MSE')
plt.title('MSE by Keypoint and Method')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# Plot improvement percentages
plt.subplot(1, 2, 2)
plt.bar(results_df['kp_id'].astype(str), results_df['baseline_to_ensemble_improvement'], color='blue', alpha=0.7, label='vs Baseline')
plt.bar(results_df['kp_id'].astype(str), results_df['weighted_to_ensemble_improvement'], color='orange', alpha=0.7, label='vs Weighted Avg')
plt.xlabel('Keypoint ID')
plt.ylabel('Improvement (%)')
plt.title('Ensemble Improvement Percentage')
plt.legend()
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Count how many times each method was the best
best_method_counts = results_df['best_method'].value_counts()
plt.figure(figsize=(8, 5))
plt.pie(best_method_counts, labels=best_method_counts.index, autopct='%1.1f%%', 
        startangle=90, colors=['gold', 'lightgreen', 'lightblue'])
plt.axis('equal')
plt.title('Best Method Distribution Across All Keypoints')
plt.show()








