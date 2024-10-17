import matplotlib.pyplot as plt
import numpy
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

file = "HBDS/Datasets/fruitohms.txt"

data = pd.read_csv(file, delimiter=' ', quotechar='"')
print(data)

def pipeline(data):

    x = data["juice"].values
    y = data["ohms"].values

    x = x.reshape(-1,1)

    def degrees(x, mn, mx):

        degrees = []
        r2s = []

        for i in range(mn, mx+1):

            poly = PolynomialFeatures(degree=i)


            X_poly = poly.fit_transform(x)

            poly_reg = LinearRegression()
            poly_reg.fit(X_poly, y)

            # Predicting
            y_pred_poly = poly_reg.predict(X_poly)

            r2 = r2_score(y, y_pred_poly)

            r2s.append(r2)
            degrees.append(i)

        plt.plot(degrees, r2s)
        plt.show()
    
    def visualize_best(degree):
        
        plt.scatter(x,y, color="orange")

        poly = PolynomialFeatures(degree)

        X_poly = poly.fit_transform(x)

        poly_reg = LinearRegression()
        poly_reg.fit(X_poly, y)

        # Predicting
        y_pred_poly = poly_reg.predict(X_poly)

        plt.plot(x, y_pred_poly)
        plt.show()

    degrees(x, 1, 20)
    visualize_best(8)

pipeline(data)