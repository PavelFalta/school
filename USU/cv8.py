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

# Load data
df = pd.read_csv(path)
print("Original data head:")
print(df.head())

# Extract all columns related to keypoint 0
columns = df.columns
target = "kp0"
target_col = [col for col in columns if target in col]
y_data = df[target_col]

print("\nTarget data head:")
print(y_data.head())

# Extract ground truth target coordinates
Y = y_data[["target_kp0_y", "target_kp0_x"]]

# Calculate baseline error using highest weight prediction
prediction_weights = y_data[["pred_kp0_val0", "pred_kp0_val1", "pred_kp0_val2", "pred_kp0_val3", "pred_kp0_val4"]]
highest_weight_idx = np.argmax(prediction_weights.values, axis=1)

# Extract the corresponding coordinates based on highest weight
baseline_pred_y = np.zeros(len(y_data))
baseline_pred_x = np.zeros(len(y_data))

for i in range(len(y_data)):
    idx = highest_weight_idx[i]
    baseline_pred_y[i] = y_data.iloc[i][f"pred_kp0_pos{idx}_y"]
    baseline_pred_x[i] = y_data.iloc[i][f"pred_kp0_pos{idx}_x"]

baseline_pred = np.column_stack((baseline_pred_y, baseline_pred_x))
baseline_mse = mean_squared_error(Y, baseline_pred)
print(f"\nBaseline MSE (using highest weight prediction): {baseline_mse:.4f}")

# Create weighted average prediction
weighted_pred_y = np.zeros(len(y_data))
weighted_pred_x = np.zeros(len(y_data))

for i in range(len(y_data)):
    weights = prediction_weights.iloc[i].values
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    
    for j in range(5):
        weighted_pred_y[i] += weights[j] * y_data.iloc[i][f"pred_kp0_pos{j}_y"]
        weighted_pred_x[i] += weights[j] * y_data.iloc[i][f"pred_kp0_pos{j}_x"]

weighted_avg_pred = np.column_stack((weighted_pred_y, weighted_pred_x))
weighted_avg_mse = mean_squared_error(Y, weighted_avg_pred)
print(f"Weighted Average MSE: {weighted_avg_mse:.4f}")

# ----------------------- IMPROVED APPROACHES -----------------------

# 1. Create more engineered features
def create_engineered_features(df):
    features = {}
    
    # Original features: all positions and weights
    for i in range(5):
        features[f'pos{i}_y'] = df[f'pred_kp0_pos{i}_y'].values
        features[f'pos{i}_x'] = df[f'pred_kp0_pos{i}_x'].values
        features[f'val{i}'] = df[f'pred_kp0_val{i}'].values
    
    # Add centroid and sigma
    features['centroid_y'] = df['pred_kp0_centroid_y'].values
    features['centroid_x'] = df['pred_kp0_centroid_x'].values
    features['sigma_y'] = df['pred_kp0_sigma_y'].values
    features['sigma_x'] = df['pred_kp0_sigma_x'].values
    
    # Add spatial relationships between predictions
    for i in range(5):
        for j in range(i+1, 5):
            # Distance between predictions
            features[f'dist_{i}_{j}'] = np.sqrt(
                (df[f'pred_kp0_pos{i}_y'] - df[f'pred_kp0_pos{j}_y'])**2 + 
                (df[f'pred_kp0_pos{i}_x'] - df[f'pred_kp0_pos{j}_x'])**2
            )
    
    # Add distances from each prediction to centroid
    for i in range(5):
        features[f'dist_to_centroid_{i}'] = np.sqrt(
            (df[f'pred_kp0_pos{i}_y'] - df['pred_kp0_centroid_y'])**2 + 
            (df[f'pred_kp0_pos{i}_x'] - df['pred_kp0_centroid_x'])**2
        )
    
    # Add normalized weights (sum to 1)
    weight_sum = df[["pred_kp0_val0", "pred_kp0_val1", "pred_kp0_val2", "pred_kp0_val3", "pred_kp0_val4"]].sum(axis=1)
    for i in range(5):
        features[f'norm_val{i}'] = df[f'pred_kp0_val{i}'].values / weight_sum
    
    # Add weight ratios
    for i in range(5):
        for j in range(i+1, 5):
            # Avoid division by zero
            ratio = df[f'pred_kp0_val{i}'] / df[f'pred_kp0_val{j}'].replace(0, 1e-10)
            features[f'weight_ratio_{i}_{j}'] = ratio
    
    # Create feature DataFrame
    return pd.DataFrame(features)

engineered_features = create_engineered_features(y_data)
print(f"\nEngineered features shape: {engineered_features.shape}")

# Scale features
scaler = StandardScaler()
X_engineered_scaled = scaler.fit_transform(engineered_features)

# Split data 
X_train, X_test, Y_train, Y_test = train_test_split(X_engineered_scaled, Y, test_size=0.2, random_state=42)

# Get baseline and weighted average predictions for test set
test_indices = np.array(range(len(Y)))[X_train.shape[0]:]
baseline_test_pred = baseline_pred[test_indices]
weighted_test_pred = weighted_avg_pred[test_indices]

# ---------- APPROACH 2: ENSEMBLE OF MODELS ----------
# Train multiple models and ensemble their predictions

# Ridge regression model - linear model good for this task
ridge_y = Ridge(alpha=0.5)
ridge_x = Ridge(alpha=0.5)

# SVR - good for capturing non-linear relationships
svr_y = SVR(kernel='rbf', C=10, epsilon=0.1)
svr_x = SVR(kernel='rbf', C=10, epsilon=0.1)

# Random Forest - good for feature importance
rf_y = RandomForestRegressor(n_estimators=100, random_state=42)
rf_x = RandomForestRegressor(n_estimators=100, random_state=42)

# Neural network - good for complex patterns
nn_y = MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', 
                    max_iter=1000, random_state=42)
nn_x = MLPRegressor(hidden_layer_sizes=(50, 25), activation='relu', 
                    max_iter=1000, random_state=42)

# Train each model on Y coordinate
ridge_y.fit(X_train, Y_train['target_kp0_y'])
svr_y.fit(X_train, Y_train['target_kp0_y'])
rf_y.fit(X_train, Y_train['target_kp0_y'])
nn_y.fit(X_train, Y_train['target_kp0_y'])

# Train each model on X coordinate
ridge_x.fit(X_train, Y_train['target_kp0_x'])
svr_x.fit(X_train, Y_train['target_kp0_x'])
rf_x.fit(X_train, Y_train['target_kp0_x'])
nn_x.fit(X_train, Y_train['target_kp0_x'])

# Make predictions
ridge_pred_y = ridge_y.predict(X_test)
svr_pred_y = svr_y.predict(X_test)
rf_pred_y = rf_y.predict(X_test)
nn_pred_y = nn_y.predict(X_test)

ridge_pred_x = ridge_x.predict(X_test)
svr_pred_x = svr_x.predict(X_test)
rf_pred_x = rf_x.predict(X_test)
nn_pred_x = nn_x.predict(X_test)

# Calculate individual model MSEs for Y coordinate
ridge_mse_y = mean_squared_error(Y_test['target_kp0_y'], ridge_pred_y)
svr_mse_y = mean_squared_error(Y_test['target_kp0_y'], svr_pred_y)
rf_mse_y = mean_squared_error(Y_test['target_kp0_y'], rf_pred_y)
nn_mse_y = mean_squared_error(Y_test['target_kp0_y'], nn_pred_y)

# Calculate individual model MSEs for X coordinate
ridge_mse_x = mean_squared_error(Y_test['target_kp0_x'], ridge_pred_x)
svr_mse_x = mean_squared_error(Y_test['target_kp0_x'], svr_pred_x)
rf_mse_x = mean_squared_error(Y_test['target_kp0_x'], rf_pred_x)
nn_mse_x = mean_squared_error(Y_test['target_kp0_x'], nn_pred_x)

print(f"\nIndividual Model MSEs:")
print(f"Ridge - Y: {ridge_mse_y:.4f}, X: {ridge_mse_x:.4f}")
print(f"SVR   - Y: {svr_mse_y:.4f}, X: {svr_mse_x:.4f}")
print(f"RF    - Y: {rf_mse_y:.4f}, X: {rf_mse_x:.4f}")
print(f"NN    - Y: {nn_mse_y:.4f}, X: {nn_mse_x:.4f}")

# Create ensemble using the best models based on individual performance
# Use inverse error weighting for ensemble
def inverse_error_weight(error):
    return 1.0 / (error + 1e-10)  # Adding small constant to avoid division by zero

# Calculate weights for Y coordinate models
y_errors = np.array([ridge_mse_y, svr_mse_y, rf_mse_y, nn_mse_y])
y_weights = inverse_error_weight(y_errors)
y_weights = y_weights / np.sum(y_weights)  # Normalize to sum to 1

# Calculate weights for X coordinate models
x_errors = np.array([ridge_mse_x, svr_mse_x, rf_mse_x, nn_mse_x])
x_weights = inverse_error_weight(x_errors)
x_weights = x_weights / np.sum(x_weights)  # Normalize to sum to 1

print(f"\nEnsemble weights for Y coordinate models: {y_weights}")
print(f"Ensemble weights for X coordinate models: {x_weights}")

# Create weighted ensemble predictions
ensemble_pred_y = (
    y_weights[0] * ridge_pred_y + 
    y_weights[1] * svr_pred_y + 
    y_weights[2] * rf_pred_y + 
    y_weights[3] * nn_pred_y
)

ensemble_pred_x = (
    x_weights[0] * ridge_pred_x + 
    x_weights[1] * svr_pred_x + 
    x_weights[2] * rf_pred_x + 
    x_weights[3] * nn_pred_x
)

ensemble_pred = np.column_stack((ensemble_pred_y, ensemble_pred_x))
ensemble_mse = mean_squared_error(Y_test, ensemble_pred)
print(f"Ensemble Model MSE: {ensemble_mse:.4f}")

# Approach 3: SVR model with automated feature selection
# SVR often works well with spatial data
svr_model = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=10, epsilon=0.1))
])

# Train separate models for Y and X coordinates
svr_model_y = svr_model
svr_model_x = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=10, epsilon=0.1))
])

# Train SVR models
svr_model_y.fit(X_train, Y_train['target_kp0_y'])
svr_model_x.fit(X_train, Y_train['target_kp0_x'])

# Make predictions
svr_pred_y = svr_model_y.predict(X_test)
svr_pred_x = svr_model_x.predict(X_test)
svr_pred = np.column_stack((svr_pred_y, svr_pred_x))

# Calculate MSE
svr_mse = mean_squared_error(Y_test, svr_pred)
print(f"SVR with Feature Engineering MSE: {svr_mse:.4f}")

# Compare all MSE values
methods = [
    "Baseline (Highest Weight)", 
    "Weighted Average",
    "SVR with Feature Engineering",
    "Model Ensemble"
]

mse_values = [
    mean_squared_error(Y_test, baseline_test_pred),
    mean_squared_error(Y_test, weighted_test_pred),
    svr_mse,
    ensemble_mse
]

# Find best method
best_method_idx = np.argmin(mse_values)
best_method = methods[best_method_idx]
best_mse = mse_values[best_method_idx]

print(f"\nBest method: {best_method} with MSE: {best_mse:.4f}")

# Plot comparison of all methods
plt.figure(figsize=(14, 10))

# Plot 1: Ground truth vs baseline
plt.subplot(2, 2, 1)
plt.scatter(Y_test['target_kp0_x'], Y_test['target_kp0_y'], 
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
plt.subplot(2, 2, 2)
plt.scatter(Y_test['target_kp0_x'], Y_test['target_kp0_y'], 
           alpha=0.5, label='Ground truth', color='blue')
plt.scatter(weighted_test_pred[:, 1], weighted_test_pred[:, 0], 
           alpha=0.5, label='Weighted average', color='green')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title(f'Ground Truth vs Weighted Average\nMSE: {mse_values[1]:.4f}')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Plot 3: Ground truth vs SVR
plt.subplot(2, 2, 3)
plt.scatter(Y_test['target_kp0_x'], Y_test['target_kp0_y'], 
           alpha=0.5, label='Ground truth', color='blue')
plt.scatter(svr_pred[:, 1], svr_pred[:, 0], 
           alpha=0.5, label='SVR', color='orange')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title(f'Ground Truth vs SVR\nMSE: {mse_values[2]:.4f}')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Plot 4: Ground truth vs ensemble
plt.subplot(2, 2, 4)
plt.scatter(Y_test['target_kp0_x'], Y_test['target_kp0_y'], 
           alpha=0.5, label='Ground truth', color='blue')
plt.scatter(ensemble_pred[:, 1], ensemble_pred[:, 0], 
           alpha=0.5, label='Ensemble', color='purple')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title(f'Ground Truth vs Ensemble\nMSE: {mse_values[3]:.4f}')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()

# Plot comparison of MSE values
plt.figure(figsize=(10, 6))
bars = plt.bar(methods, mse_values, color=['red', 'green', 'orange', 'purple'])
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Comparison of Methods')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Highlight the best method
bars[best_method_idx].set_color('gold')
plt.annotate(f'Best: {best_mse:.4f}', 
             xy=(best_method_idx, best_mse),
             xytext=(best_method_idx, best_mse + 1),
             ha='center',
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()








