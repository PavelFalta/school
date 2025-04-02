path = "data/data-recovery.csv"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor



df = pd.read_csv(path)
print("Original data head:")
print(df.head())

columns = df.columns
target = "kp0"
target_col = [col for col in columns if target in col]
y_data = df[target_col]

print("\nTarget data head:")
print(y_data.head())


Y = y_data[["target_kp0_y", "target_kp0_x"]]
X = y_data.drop(columns=["target_kp0_y", "target_kp0_x", 
                         "pred_kp0_centroid_y", "pred_kp0_centroid_x", 
                         "pred_kp0_sigma_y", "pred_kp0_sigma_x"])

print("\nFeatures head:")
print(X.head())

mse = mean_squared_error(y_data[["pred_kp0_pos0_y", "pred_kp0_pos0_x"]], y_data[["target_kp0_y", "target_kp0_x"]])
print(f"\nMSE between pred_kp0_pos0_y and target_kp0_y: {mse:.4f}")


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)


Y_pred = model.predict(X_test)


mse = mean_squared_error(Y_test, Y_pred)
print(f"\nMean Squared Error: {mse:.4f}")


plt.figure(figsize=(10, 10))
plt.scatter(Y_test['target_kp0_x'], Y_test['target_kp0_y'], 
           alpha=0.5, label='True points', color='blue')
plt.scatter(Y_pred[:, 1], Y_pred[:, 0], 
           alpha=0.5, label='Predicted points', color='red')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('True vs Predicted Keypoint Positions')
plt.legend()
plt.grid(True)
plt.axis('equal')  
plt.show()








