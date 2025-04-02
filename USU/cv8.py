import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import re
import os
import time
from tqdm import tqdm
from warnings import filterwarnings

filterwarnings("ignore")


def nacti_data(cesta):
    print(f"Načítání dat z {cesta}...")
    return pd.read_csv(cesta)


def najdi_body(df):
    vsechny_sloupce = df.columns
    bod_regex = re.compile(r'target_kp(\d+)_[xy]')
    body_shody = [bod_regex.match(sloupec) for sloupec in vsechny_sloupce]
    cisla_bodu = sorted(list(set([int(shoda.group(1)) for shoda in body_shody if shoda])))
    print(f"Nalezeno {len(cisla_bodu)} bodů: {cisla_bodu}")
    return cisla_bodu


def rozdel_data(df, test_velikost=0.2):
    radky_indexy = np.arange(len(df))
    train_idx, test_idx = train_test_split(radky_indexy, test_size=test_velikost, random_state=42)
    print(f"Data rozdělena: {len(train_idx)} trénovacích vzorků, {len(test_idx)} testovacích vzorků")
    return train_idx, test_idx


def udel_zakladni_predikce(df, cislo_bodu):
    prefix = f"kp{cislo_bodu}"
    sloupce_vah = [f'pred_{prefix}_val{i}' for i in range(5)]
    idx_max_vahy = np.argmax(df.loc[:, sloupce_vah].values, axis=1)
    
    y_zaklad = np.zeros(len(df))
    x_zaklad = np.zeros(len(df))
    
    for i in range(len(df)):
        idx = idx_max_vahy[i]
        y_zaklad[i] = df.iloc[i][f'pred_{prefix}_pos{idx}_y']
        x_zaklad[i] = df.iloc[i][f'pred_{prefix}_pos{idx}_x']
    
    return y_zaklad, x_zaklad


def udel_priznaky(df, cislo_bodu):
    prefix = f"kp{cislo_bodu}"
    sloupce_priznaku = []
    
    #tady beru predikce pozic
    for i in range(5):
        sloupce_priznaku.extend([f'pred_{prefix}_pos{i}_y', f'pred_{prefix}_pos{i}_x'])
    
    #sem jdou vahy
    sloupce_priznaku.extend([f'pred_{prefix}_val{i}' for i in range(5)])
    
    #centroid a sigma nakonec
    sloupce_priznaku.extend([
        f'pred_{prefix}_centroid_y', f'pred_{prefix}_centroid_x',
        f'pred_{prefix}_sigma_y', f'pred_{prefix}_sigma_x'
    ])
    
    return sloupce_priznaku


def trenuj_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def zpracuj_bod(df, cislo_bodu, train_idx, test_idx):
    prefix = f"kp{cislo_bodu}"
    
    #najdu cilovy sloupecky
    y_sloupec = f"target_{prefix}_y"
    x_sloupec = f"target_{prefix}_x"
    
    #ulozim si skutecny hodnoty
    y_pravda = df.loc[test_idx, y_sloupec].values
    x_pravda = df.loc[test_idx, x_sloupec].values
    
    #tady ziskam zakladni predikce
    y_zaklad, x_zaklad = udel_zakladni_predikce(df, cislo_bodu)
    
    #vytvori a oskaluje priznaky
    sloupce_priznaku = udel_priznaky(df, cislo_bodu)
    X = df[sloupce_priznaku].values
    skalovac = StandardScaler()
    X_skalovane = skalovac.fit_transform(X)
    
    X_train = X_skalovane[train_idx]
    X_test = X_skalovane[test_idx]
    
    y_train_y = df.loc[train_idx, y_sloupec].values
    y_train_x = df.loc[train_idx, x_sloupec].values
    
    #trenuju modely
    y_model = trenuj_model(X_train, y_train_y)
    x_model = trenuj_model(X_train, y_train_x)
    
    #tedka udelam predikce
    y_pred_rf = y_model.predict(X_test)
    x_pred_rf = x_model.predict(X_test)
    
    #vratim vysledky v slovniku
    return {
        'y_pravda': y_pravda,
        'x_pravda': x_pravda,
        'y_zaklad': y_zaklad[test_idx],
        'x_zaklad': x_zaklad[test_idx],
        'y_rf': y_pred_rf,
        'x_rf': x_pred_rf,
        'y_model': y_model,
        'x_model': x_model
    }


def spocti_mse(predikce):
    mse_zaklad = mean_squared_error(
        np.column_stack((predikce['y_pravda'], predikce['x_pravda'])),
        np.column_stack((predikce['y_zaklad'], predikce['x_zaklad']))
    )
    
    mse_rf = mean_squared_error(
        np.column_stack((predikce['y_pravda'], predikce['x_pravda'])),
        np.column_stack((predikce['y_rf'], predikce['x_rf']))
    )
    
    return mse_zaklad, mse_rf


def spocti_chyby(vysledky, cisla_bodu, test_idx):
    pravdive_hodnoty = pd.DataFrame(index=test_idx)
    zakladni_hodnoty = pd.DataFrame(index=test_idx)
    rf_hodnoty = pd.DataFrame(index=test_idx)
    chyby_zaklad = pd.DataFrame(index=test_idx)
    chyby_rf = pd.DataFrame(index=test_idx)
    
    for cislo in cisla_bodu:
        prefix = f"kp{cislo}"
        predikce = vysledky[cislo]
        
        #ulozim hodnoty do dataframu
        pravdive_hodnoty[f'{prefix}_y'] = predikce['y_pravda']
        pravdive_hodnoty[f'{prefix}_x'] = predikce['x_pravda']
        
        zakladni_hodnoty[f'{prefix}_y'] = predikce['y_zaklad']
        zakladni_hodnoty[f'{prefix}_x'] = predikce['x_zaklad']
        
        rf_hodnoty[f'{prefix}_y'] = predikce['y_rf']
        rf_hodnoty[f'{prefix}_x'] = predikce['x_rf']
        
        #vypocitam chyby pro Y souradnice
        chyby_zaklad[f'{prefix}'] = np.abs(predikce['y_pravda'] - predikce['y_zaklad'])
        chyby_rf[f'{prefix}'] = np.abs(predikce['y_pravda'] - predikce['y_rf'])
    
    #vypocet celkovy mse, docela slozity
    def spocti_celkove_mse(pravda_df, predikce_df):
        hodnoty_pravda = []
        hodnoty_predikce = []
        
        for cislo in cisla_bodu:
            prefix = f"kp{cislo}"
            hodnoty_pravda.append(pravda_df[[f'{prefix}_y', f'{prefix}_x']].values)
            hodnoty_predikce.append(predikce_df[[f'{prefix}_y', f'{prefix}_x']].values)
        
        hodnoty_pravda = np.concatenate(hodnoty_pravda, axis=1)
        hodnoty_predikce = np.concatenate(hodnoty_predikce, axis=1)
        
        return mean_squared_error(hodnoty_pravda, hodnoty_predikce)
    
    celkova_mse_zaklad = spocti_celkove_mse(pravdive_hodnoty, zakladni_hodnoty)
    celkova_mse_rf = spocti_celkove_mse(pravdive_hodnoty, rf_hodnoty)
    zlepseni_procenta = 100 * (celkova_mse_zaklad - celkova_mse_rf) / celkova_mse_zaklad
    
    print(f"\nCelkové výsledky:")
    print(f"MSE bodu s největší vahou: {celkova_mse_zaklad:.4f}")
    print(f"MSE náhodného lesa: {celkova_mse_rf:.4f}")
    print(f"Celkové zlepšení: {zlepseni_procenta:.2f}%")
    
    return pravdive_hodnoty, zakladni_hodnoty, rf_hodnoty, chyby_zaklad, chyby_rf


def uloz_predikce(pravdive_hodnoty, zakladni_hodnoty, rf_hodnoty, chyby_zaklad, chyby_rf, cisla_bodu, cesta_vystup):
    print(f"Ukládání predikcí do {cesta_vystup}...")
    vysledky_df = pd.DataFrame(index=pravdive_hodnoty.index)
    
    for cislo in cisla_bodu:
        prefix = f"kp{cislo}"
        
        #beru jen y souradnice a chyby pro prehlednost
        vysledky_df[f'skutecna_{prefix}_y'] = pravdive_hodnoty[f'{prefix}_y']
        vysledky_df[f'nejvetsi_vaha_{prefix}_y'] = zakladni_hodnoty[f'{prefix}_y']
        vysledky_df[f'predikce_{prefix}_y'] = rf_hodnoty[f'{prefix}_y']
        vysledky_df[f'chyba_nejvetsi_vaha_{prefix}'] = chyby_zaklad[f'{prefix}']
        vysledky_df[f'chyba_predikce_{prefix}'] = chyby_rf[f'{prefix}']
    
    #vytvorim adresar kdyztak
    os.makedirs(os.path.dirname(cesta_vystup), exist_ok=True)
    vysledky_df.to_csv(cesta_vystup)
    print(f"Predikce uloženy do {cesta_vystup}")


def predikuj_body(cesta_vstup="data/data-recovery.csv", cesta_vystup="data/predikce_bodu.csv"):
    cas_start = time.time()
    
    #nactu data z csv
    df = nacti_data(cesta_vstup)
    
    #najdu vsechny body
    cisla_bodu = najdi_body(df)
    
    #rozdelim data na trenink a test
    train_idx, test_idx = rozdel_data(df)
    
    #zpracuju kazdy bod zvlast
    print("Zpracování bodů...")
    vysledky = {}
    for cislo in tqdm(cisla_bodu, desc="Zpracování bodů"):
        vysledky[cislo] = zpracuj_bod(df, cislo, train_idx, test_idx)
    
    #spocitam chyby a tak
    pravdive_hodnoty, zakladni_hodnoty, rf_hodnoty, chyby_zaklad, chyby_rf = spocti_chyby(
        vysledky, cisla_bodu, test_idx
    )
    
    #ulozim vysledky do souboru
    uloz_predikce(pravdive_hodnoty, zakladni_hodnoty, rf_hodnoty, chyby_zaklad, chyby_rf, cisla_bodu, cesta_vystup)
    
    cas_konec = time.time()
    print(f"Celková doba zpracování: {cas_konec - cas_start:.2f} sekund")
    
    return vysledky


if __name__ == "__main__":
    predikuj_body()