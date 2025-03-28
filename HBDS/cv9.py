import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit
from pmdarima import auto_arima
from sklearn.ensemble import VotingRegressor

train_path = 'HBDS/Datasets/DailyDelhiClimateTrain.csv'
test_path = 'HBDS/Datasets/DailyDelhiClimateTest.csv'

predict_len = 7

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(train_df.head())
print(train_df.info())

time_series = train_df.copy()
time_series['date'] = pd.to_datetime(train_df['date'])
time_series.drop(columns=['humidity', 'wind_speed', 'meanpressure'], inplace=True)
time_series.set_index('date', inplace=True)

test_series = test_df.copy()
test_series['date'] = pd.to_datetime(test_df['date'])
test_series.drop(columns=['humidity', 'wind_speed', 'meanpressure'], inplace=True)
test_series.set_index('date', inplace=True)

print(time_series.head())
print(test_series.head())


class SarimaRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_ = None

    def fit(self, X, y):
        self.model_ = SARIMAX(y, order=self.order, seasonal_order=self.seasonal_order)
        self.model_ = self.model_.fit(disp=False)
        return self

    def predict(self, X):
        forecast = self.model_.get_forecast(steps=len(X))
        return forecast.predicted_mean
    

X_train = np.arange(len(time_series)).reshape(-1, 1)
y_train = time_series.values


X_test = np.arange(len(time_series), len(time_series) + predict_len).reshape(-1, 1)  


sarima_model = SarimaRegressor(order=(1, 1, 2), seasonal_order=(0, 0, 0, 12))
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)


voting_model = VotingRegressor(estimators=[
    ('sarima', sarima_model),
    ('rf', rf_model),
    ('gb', gb_model)
])


voting_model.fit(X_train, y_train)


voting_forecast = voting_model.predict(X_test)


rmse = np.sqrt(mean_squared_error(test_series[:predict_len], voting_forecast))  
print(f"VotingRegressor RMSE: {rmse}")


plt.figure(figsize=(12, 6))
plt.plot(time_series.index, time_series, label="Train Data")
plt.plot(test_series.index[:predict_len], test_series[:predict_len], label="Test Data", color='orange')  
plt.plot(test_series.index[:predict_len], voting_forecast, label="Voting Forecast", color='green')  
plt.title("Voting Regressor Model")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.legend()
plt.show()
class SarimaWrapper:
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model_ = None

    def fit(self, train):
        self.model_ = SARIMAX(train, order=self.order, seasonal_order=self.seasonal_order)
        self.model_ = self.model_.fit(disp=False)

    def predict(self, steps):
        forecast = self.model_.get_forecast(steps=steps)
        return forecast.predicted_mean

    def residuals(self, train):
        fitted_values = self.model_.fittedvalues
        return train - fitted_values
    

time_series = time_series['meantemp']
train = time_series
test = test_series['meantemp'][:predict_len]


sarima = SarimaWrapper(order=(1, 1, 1), seasonal_order=(1, 0, 1, 12))
sarima.fit(train)
sarima_forecast = sarima.predict(len(test))


residuals_train = sarima.residuals(train)


X_train = np.arange(len(residuals_train)).reshape(-1, 1)
y_train = residuals_train

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)


X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
rf_residuals = rf_model.predict(X_test)
gb_residuals = gb_model.predict(X_test)


hybrid_forecast_rf = sarima_forecast + rf_residuals
hybrid_forecast_gb = sarima_forecast + gb_residuals


rmse_rf = np.sqrt(mean_squared_error(test, hybrid_forecast_rf))
rmse_gb = np.sqrt(mean_squared_error(test, hybrid_forecast_gb))

print(f"Hybrid Model (SARIMA + Random Forest) RMSE: {rmse_rf}")
print(f"Hybrid Model (SARIMA + Gradient Boosting) RMSE: {rmse_gb}")


plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label="Train Data")
plt.plot(test.index, test, label="Test Data", color='orange')
plt.plot(test.index, hybrid_forecast_rf, label="Hybrid Forecast (Random Forest)", color='green')
plt.plot(test.index, hybrid_forecast_gb, label="Hybrid Forecast (Gradient Boosting)", color='purple')
plt.title("Hybrid Model Forecast")
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.legend()
plt.show()
