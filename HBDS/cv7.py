import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from keras import layers, Sequential
import math

path = 'HBDS/Datasets/Student_Performance.csv'
df = pd.read_csv(path)

print(df.head())
print(df.info())


X = df.drop(['Performance Index'], axis=1)  # Features
y = df['Performance Index']  # Target variable

# Define categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Apply Label Encoding to categorical columns
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])  # Transform the column
    label_encoders[col] = le  # Save the encoder for later use if needed

# Normalize numeric columns
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

print("\nLinear Regression:")
print("Mean Squared Error:", mean_squared_error(y_test, lr_predictions))
print("Root Mean Squared Error:", math.sqrt(mean_squared_error(y_test, lr_predictions)))
print("R² Score:", r2_score(y_test, lr_predictions))

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

print("\nRandom Forest Regression:")
print("Mean Squared Error:", mean_squared_error(y_test, rf_predictions))
print("Root Mean Squared Error:", math.sqrt(mean_squared_error(y_test, rf_predictions)))
print("R² Score:", r2_score(y_test, rf_predictions))

gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)

print("\nGradient Boosting Regression:")
print("Mean Squared Error:", mean_squared_error(y_test, gb_predictions))
print("Root Mean Squared Error:", math.sqrt(mean_squared_error(y_test, gb_predictions)))
print("R² Score:", r2_score(y_test, gb_predictions))

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

print("\nDecision Tree Regression:")
print("Mean Squared Error:", mean_squared_error(y_test, dt_predictions))
print("Root Mean Squared Error:", math.sqrt(mean_squared_error(y_test, dt_predictions)))
print("R² Score:", r2_score(y_test, dt_predictions))

base_models = [
    ('Linear Regression', lr_model),
    ('Random Forest', rf_model),
    ('Gradient Boosting', gb_model),
    ('Decision Tree', dt_model)
]

weights = [0.2, 0.4, 0.3, 0.1]

# Collect predictions from base models
base_model_predictions = []
for name, model in base_models:
    preds = model.predict(X_test)  # Predictions on test set
    base_model_predictions.append(preds)

# Weighted majority voting: weighted average of predictions
weighted_predictions = np.zeros_like(base_model_predictions[0])
for i, preds in enumerate(base_model_predictions):
    weighted_predictions += weights[i] * preds

print("\nMajority Voting Regression model:")
print("Mean Squared Error:", mean_squared_error(y_test, weighted_predictions))
print("Root Mean Squared Error:", math.sqrt(mean_squared_error(y_test, weighted_predictions)))
print("R² Score:", r2_score(y_test, weighted_predictions))

# Train base models and collect predictions on training data
meta_features_train = np.zeros((X_train.shape[0], len(base_models)))
meta_features_test = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models):
    model.fit(X_train, y_train)  # Train base model
    meta_features_train[:, i] = model.predict(X_train)  # Predictions on training set
    meta_features_test[:, i] = model.predict(X_test)  # Predictions on test set

# Define the neural network
meta_model = Sequential([
    layers.Dense(32, input_dim=meta_features_train.shape[1], activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)  # Single output for regression
])

# Compile the model
meta_model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Train the meta-model
meta_model.fit(meta_features_train, y_train, epochs=50, batch_size=32, verbose=0)


# Make predictions using the stacking model
stacking_predictions = meta_model.predict(meta_features_test)

print("\nMajority Voting Regression model:")
print("Mean Squared Error:", mean_squared_error(y_test, stacking_predictions))
print("Root Mean Squared Error:", math.sqrt(mean_squared_error(y_test, stacking_predictions)))
print("R² Score:", r2_score(y_test, stacking_predictions))