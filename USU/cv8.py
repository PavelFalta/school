path = "data/data-recovery.csv"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

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
# Find which prediction has the highest weight for each sample
prediction_weights = y_data[["pred_kp0_val0", "pred_kp0_val1", "pred_kp0_val2", "pred_kp0_val3", "pred_kp0_val4"]]
highest_weight_idx = np.argmax(prediction_weights.values, axis=1)

# Extract the corresponding coordinates based on highest weight
baseline_pred_y = np.zeros(len(y_data))
baseline_pred_x = np.zeros(len(y_data))

for i in range(len(y_data)):
    idx = highest_weight_idx[i]
    baseline_pred_y[i] = y_data.iloc[i][f"pred_kp0_pos{idx}_y"]
    baseline_pred_x[i] = y_data.iloc[i][f"pred_kp0_pos{idx}_x"]

# Calculate baseline MSE
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
    
    # Calculate weighted average coordinates
    for j in range(5):  # 5 predictions (0-4)
        weighted_pred_y[i] += weights[j] * y_data.iloc[i][f"pred_kp0_pos{j}_y"]
        weighted_pred_x[i] += weights[j] * y_data.iloc[i][f"pred_kp0_pos{j}_x"]

# Calculate weighted average MSE
weighted_avg_pred = np.column_stack((weighted_pred_y, weighted_pred_x))
weighted_avg_mse = mean_squared_error(Y, weighted_avg_pred)
print(f"Weighted Average MSE: {weighted_avg_mse:.4f}")

# Prepare features for the machine learning model
# Use all predictions and their weights as features
feature_cols = []
for i in range(5):
    feature_cols.extend([f"pred_kp0_pos{i}_y", f"pred_kp0_pos{i}_x", f"pred_kp0_val{i}"])

# Also add centroid and sigma information
feature_cols.extend(["pred_kp0_centroid_y", "pred_kp0_centroid_x", "pred_kp0_sigma_y", "pred_kp0_sigma_x"])

X = y_data[feature_cols]

print("\nFeatures head:")
print(X.head())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)

# Get the baseline and weighted average predictions for the test set
test_indices = np.array(range(len(Y)))[X_train.shape[0]:]
baseline_test_pred = baseline_pred[test_indices]
weighted_test_pred = weighted_avg_pred[test_indices]

# Train random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Calculate MSE
rf_mse = mean_squared_error(Y_test, rf_pred)
print(f"Random Forest MSE: {rf_mse:.4f}")

# Plot results
plt.figure(figsize=(15, 5))

# Plot 1: Compare ground truth with baseline (highest weight)
plt.subplot(1, 3, 1)
plt.scatter(Y_test['target_kp0_x'], Y_test['target_kp0_y'], 
           alpha=0.5, label='Ground truth', color='blue')
plt.scatter(baseline_test_pred[:, 1], baseline_test_pred[:, 0], 
           alpha=0.5, label='Highest weight', color='red')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title(f'Ground Truth vs Highest Weight\nMSE: {mean_squared_error(Y_test, baseline_test_pred):.4f}')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Plot 2: Compare ground truth with weighted average
plt.subplot(1, 3, 2)
plt.scatter(Y_test['target_kp0_x'], Y_test['target_kp0_y'], 
           alpha=0.5, label='Ground truth', color='blue')
plt.scatter(weighted_test_pred[:, 1], weighted_test_pred[:, 0], 
           alpha=0.5, label='Weighted average', color='green')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title(f'Ground Truth vs Weighted Average\nMSE: {mean_squared_error(Y_test, weighted_test_pred):.4f}')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Plot 3: Compare ground truth with random forest
plt.subplot(1, 3, 3)
plt.scatter(Y_test['target_kp0_x'], Y_test['target_kp0_y'], 
           alpha=0.5, label='Ground truth', color='blue')
plt.scatter(rf_pred[:, 1], rf_pred[:, 0], 
           alpha=0.5, label='Random Forest', color='purple')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title(f'Ground Truth vs Random Forest\nMSE: {rf_mse:.4f}')
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.tight_layout()
plt.show()

# Feature importance
if hasattr(rf_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance.head(10))
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'][:10], feature_importance['Importance'][:10])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()
    plt.show()








