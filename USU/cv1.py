#generování dat
import numpy as np
import pandas as pd
from numpy.random import randn
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
df.head(10)