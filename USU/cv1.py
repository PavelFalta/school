#generování dat
import numpy as np
import pandas as pd
from numpy.random import randn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

N = 1000 # počet datových bodů

# pomocí gausovského rozdělení nagenerujeme body v prostoru váha-výška
# generování váhy v kg
def generuj_vahu(vaha_prumer = 80, vaha_sigma = 12, kolik = 100):
  vaha = vaha_prumer+randn(kolik)*vaha_sigma
  vaha[vaha < vaha_prumer-4*vaha_sigma] = vaha_prumer-4*vaha_sigma # orezani nepravdepodobnych hodnot - podvaha
  vaha[vaha > vaha_prumer+4*vaha_sigma] = vaha_prumer+4*vaha_sigma # orezani nepravdepodobnych hodnot - nadpodvaha
  return vaha

#generování výšky v cm
def generuj_vysku(vyska_prumer = 180, vyska_sigma = 15, kolik = 100):
  vyska = vyska_prumer+randn(kolik)*vyska_sigma
  vyska[vyska < vyska_prumer-4*vyska_sigma] = vyska_prumer-4*vyska_sigma
  vyska[vyska > vyska_prumer+4*vyska_sigma] = vyska_prumer+4*vyska_sigma
  return vyska

# spocteni body mass indexu BMI
vaha = generuj_vahu(kolik=N)
vyska = generuj_vysku(kolik=N)
bmi = vaha/(vyska/100)**2
data = {"vyska": vyska ,"vaha" : vaha, "bmi" : bmi}
df = pd.DataFrame(data) #
df.to_csv('data_lide.csv', index = False)
print(df.head(10))
# df['m^2'] = df['vyska']**2
# df['vaha*vyska'] = df['vyska']*df['vaha']
# df['h^2'] = df['vaha']**2


# plt.scatter(data['vyska'], data['vaha'], marker = 'x')
# plt.title("Datové body")
# plt.xlabel('výška [m]')
# plt.ylabel('váha [kg]')
# plt.legend()
# plt.show()

promichano_index = df.index.to_list()
np.random.shuffle(promichano_index)

trenovaci_data_velikost = int(len(df)*0.80) # vezmeme 80 % pro nauceni modelu
print(df.index[10])
trenovaci_data = df.filter(promichano_index[:trenovaci_data_velikost], axis = 0) # vem nahodne indexy
testovaci_data = df.filter(promichano_index[trenovaci_data_velikost:], axis = 0) # vem nahodne indexy

# plt.scatter(trenovaci_data['vyska'], trenovaci_data['vaha'], marker = 'x', label='trénovací')
# plt.scatter(testovaci_data['vyska'], testovaci_data['vaha'], marker = 'o', label='testovací')
# plt.title("Datové body")
# plt.xlabel('výška [m]')
# plt.ylabel('váha [kg]')
# plt.legend()
# plt.show()

# statisticke ukazatele
print(trenovaci_data.describe())
print(testovaci_data.describe())

import numpy.linalg as la

#priprava dat pro linearni regresi
y = trenovaci_data['bmi']
X_t = np.array((np.ones(y.shape), trenovaci_data['vyska'], trenovaci_data['vaha']))
X = X_t.transpose()

# sestaveni matice a prave strany
A = X_t @ X # np.dot(X^T,X)
b = X_t @ y
# vypocet koeficientu resenim soustavy lin. rovnic
koeficienty = la.solve(A,b)
bmi_hat_trenovaci = X @ koeficienty # vypocet predikce na trenovacich datech


X_test_t = np.array(( np.ones(len(testovaci_data)),testovaci_data['vyska'], testovaci_data['vaha']))
X_test = X_test_t.transpose()
bmi_hat_testovaci = X_test @ koeficienty

# vykresleni predikce a reality pro trenovaci data
ax = plt.axes(projection = '3d')
ax.scatter3D(trenovaci_data['vyska'], trenovaci_data['vaha'], trenovaci_data['bmi'], label = 'realita');
ax.scatter3D(trenovaci_data['vyska'], trenovaci_data['vaha'], bmi_hat_trenovaci, label = 'predikce');
ax.legend()
ax.set_xlabel('výška [m]')
ax.set_ylabel('váha [kg]')
ax.set_zlabel('bmi')


# vypocet chyby
mse_ls_modelu_trenovaci = ((trenovaci_data['bmi']-bmi_hat_trenovaci)**2).mean()
mse_ls_modelu_testovaci = ((testovaci_data['bmi']-bmi_hat_testovaci)**2).mean()

print(f"Chyba na trenovacich datech {mse_ls_modelu_trenovaci}")
print(f"Chyba na testovacich datech {mse_ls_modelu_testovaci}\n\n")

#import polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


# sestavení regresorů z trénovacích dat
X = np.array((trenovaci_data['vyska'], trenovaci_data['vaha'])).transpose()
y = trenovaci_data['bmi']
# vytvoření a naučení modelu - celá věda je toto

best = (0,0)

for i in range(1,10):
    print(f"Polynomial features + linear regression, polynom {i}")
    poly_model = make_pipeline(PolynomialFeatures(i), LinearRegression())
    poly_model.fit(X, y)

    # predikce na trénovacích datech
    bmi_hat_trenovaci = poly_model.predict(X)
    # predikce na testovacích datech
    X_test = np.array((testovaci_data['vyska'], testovaci_data['vaha'])).transpose()
    bmi_hat_testovaci = poly_model.predict(X_test)

    if best[0] == 0 or best[0] > ((testovaci_data['bmi']-bmi_hat_testovaci)**2).mean():
        best = (((testovaci_data['bmi']-bmi_hat_testovaci)**2).mean(), i)


print(f"Nejlepší model je polynom {best[1]} stupně")
poly_model = make_pipeline(PolynomialFeatures(best[1]), LinearRegression())
poly_model.fit(X, y)

#print how the polynomial model looks like
print(poly_model.named_steps['linearregression'].coef_)
print(poly_model.named_steps['linearregression'].intercept_)

# predikce na trénovacích datech
bmi_hat_trenovaci = poly_model.predict(X)
# predikce na testovacích datech
X_test = np.array((testovaci_data['vyska'], testovaci_data['vaha'])).transpose()
bmi_hat_testovaci = poly_model.predict(X_test)

# vykresleni predikce a reality pro trenovaci data
ax = plt.axes(projection = '3d')
ax.scatter3D(trenovaci_data['vyska'], trenovaci_data['vaha'], trenovaci_data['bmi'], label = 'realita');
ax.scatter3D(trenovaci_data['vyska'], trenovaci_data['vaha'], bmi_hat_trenovaci, label = 'predikce');
ax.legend()
ax.set_xlabel('výška [m]')
ax.set_ylabel('váha [kg]')
ax.set_zlabel('bmi')

# vypocet chyby
mse_ls_modelu_trenovaci = ((trenovaci_data['bmi']-bmi_hat_trenovaci)**2).mean()
mse_ls_modelu_testovaci = ((testovaci_data['bmi']-bmi_hat_testovaci)**2).mean()

print(f"Polynomial features + logistic regression")
print(f"Chyba na trenovacich datech {mse_ls_modelu_trenovaci}")
print(f"Chyba na testovacich datech {mse_ls_modelu_testovaci}")

plt.show()