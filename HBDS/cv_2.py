import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

url = 'HBDS?Datasets/decathlon.csv'
data = pd.read_csv(url)

print(data.head())

msno.bar(data)
plt.grid()
# plt.show()

data = data.drop(columns=["Unnamed: 0"])

corr = data.corr()


plt.figure(figsize=(10,8), dpi =500)
sns.set_theme(font_scale=0.18) 
sns.heatmap(corr,annot=True,fmt=".2f", linewidth=.5)

# plt.show()

complete_cases = data.dropna()

X_complete = complete_cases.drop(columns=['Points','High.jump'])
y_complete = complete_cases['High.jump']

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_complete = poly.fit_transform(X_complete)

poly_features = poly.get_feature_names_out(['X100m',  'Long.jump',  'Shot.put',  'X400m',  'X110m.hurdle',  'Discus',  'Pole.vault' , 'Javeline',  'X1500m'])
X_poly_complete_df = pd.DataFrame(X_poly_complete, columns=poly_features)


model_complete = LinearRegression()
model_complete.fit(X_poly_complete_df, y_complete)

y_pred_complete = model_complete.predict(X_poly_complete_df)
r2_complete = r2_score(y_complete, y_pred_complete)
mse_complete = mean_squared_error(y_complete, y_pred_complete)

print(f'R^2 (Complete cases): {r2_complete}')
print(f'MSE (Complete cases): {mse_complete}')


knn_imputer = KNNImputer(n_neighbors=2)
knn_imputed = knn_imputer.fit_transform(data)
knn_imputed_df = pd.DataFrame(knn_imputed, columns=data.columns)

# Define the features (X) and target variable (y)
X = knn_imputed_df.drop(columns=['Points', 'High.jump'])
y = knn_imputed_df['High.jump']

# Add polynomial features for second-degree polynomial regression
# Use PolynomialFeatures to create polynomial terms up to degree 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Convert the polynomial features into a DataFrame with readable column names
poly_features = poly.get_feature_names_out(['X100m',  'Long.jump',  'Shot.put', 'X400m',  'X110m.hurdle',  'Discus',  'Pole.vault' , 'Javeline',  'X1500m'])
X_poly_df = pd.DataFrame(X_poly, columns=poly_features)

# Fit a linear regression model using the polynomial features
model = LinearRegression()
model.fit(X_poly_df, y)

# Predict the values and calculate the R^2 score
y_pred = model.predict(X_poly_df)
r2_knn = r2_score(y, y_pred)
mse_knn = mean_squared_error(y, y_pred)
print(f'R^2 (KNN method application): {r2_knn}')
print(f'MSE (KNN method application): {mse_knn}')