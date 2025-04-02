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


def nacist_data(cesta):
    """Načte data z CSV souboru."""
    print(f"Načítání dat z {cesta}...")
    return pd.read_csv(cesta)


def najit_body(df):
    """Najde všechny klíčové body v datasetu pomocí regex."""
    vsechny_sloupce = df.columns
    vzor_bodu = re.compile(r'target_kp(\d+)_[xy]')
    shody_bodu = [vzor_bodu.match(sloupec) for sloupec in vsechny_sloupce]
    id_bodu = sorted(list(set([int(shoda.group(1)) for shoda in shody_bodu if shoda])))
    print(f"Nalezeno {len(id_bodu)} bodů: {id_bodu}")
    return id_bodu


def rozdelit_data(df, test_velikost=0.2):
    """Rozdělí data na trénovací a testovací sadu."""
    indexy_radku = np.arange(len(df))
    trenovaci_indexy, testovaci_indexy = train_test_split(indexy_radku, test_size=test_velikost, random_state=42)
    print(f"Data rozdělena: {len(trenovaci_indexy)} trénovacích vzorků, {len(testovaci_indexy)} testovacích vzorků")
    return trenovaci_indexy, testovaci_indexy


def vytvorit_zakladni_predikce(df, id_bod):
    """Vytvoří predikce na základě bodu s nejvyšší vahou."""
    prefix_bodu = f"kp{id_bod}"
    vahy_sloupce = [f'pred_{prefix_bodu}_val{i}' for i in range(5)]
    indexy_nejvyssi_vahy = np.argmax(df.loc[:, vahy_sloupce].values, axis=1)
    
    zakladni_y = np.zeros(len(df))
    zakladni_x = np.zeros(len(df))
    
    for i in range(len(df)):
        idx = indexy_nejvyssi_vahy[i]
        zakladni_y[i] = df.iloc[i][f'pred_{prefix_bodu}_pos{idx}_y']
        zakladni_x[i] = df.iloc[i][f'pred_{prefix_bodu}_pos{idx}_x']
    
    return zakladni_y, zakladni_x


def vytvorit_priznaky(df, id_bod):
    """Vytvoří příznakové vektory pro daný bod."""
    prefix_bodu = f"kp{id_bod}"
    priznaky_sloupce = []
    
    #tady beru predikce pozic
    for i in range(5):
        priznaky_sloupce.extend([f'pred_{prefix_bodu}_pos{i}_y', f'pred_{prefix_bodu}_pos{i}_x'])
    
    #sem jdou vahy
    priznaky_sloupce.extend([f'pred_{prefix_bodu}_val{i}' for i in range(5)])
    
    #centroid a sigma nakonec
    priznaky_sloupce.extend([
        f'pred_{prefix_bodu}_centroid_y', f'pred_{prefix_bodu}_centroid_x',
        f'pred_{prefix_bodu}_sigma_y', f'pred_{prefix_bodu}_sigma_x'
    ])
    
    return priznaky_sloupce


def trenovat_model(X_trenovaci, y_trenovaci):
    """Natrénuje model Random Forest."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_trenovaci, y_trenovaci)
    return model


def zpracovat_bod(df, id_bod, trenovaci_indexy, testovaci_indexy):
    """Zpracuje jeden klíčový bod - trénování a predikce."""
    prefix_bodu = f"kp{id_bod}"
    
    #najdu cilovy sloupecky
    sloupec_y = f"target_{prefix_bodu}_y"
    sloupec_x = f"target_{prefix_bodu}_x"
    
    #ulozim si skutecny hodnoty
    skutecne_y = df.loc[testovaci_indexy, sloupec_y].values
    skutecne_x = df.loc[testovaci_indexy, sloupec_x].values
    
    #tady ziskam zakladni predikce
    zakladni_y, zakladni_x = vytvorit_zakladni_predikce(df, id_bod)
    
    #vytvori a oskaluje priznaky
    priznaky_sloupce = vytvorit_priznaky(df, id_bod)
    X = df[priznaky_sloupce].values
    skaler = StandardScaler()
    X_skalovane = skaler.fit_transform(X)
    
    X_trenovaci = X_skalovane[trenovaci_indexy]
    X_testovaci = X_skalovane[testovaci_indexy]
    
    y_trenovaci_y = df.loc[trenovaci_indexy, sloupec_y].values
    y_trenovaci_x = df.loc[trenovaci_indexy, sloupec_x].values
    
    #trenuju modely
    model_y = trenovat_model(X_trenovaci, y_trenovaci_y)
    model_x = trenovat_model(X_trenovaci, y_trenovaci_x)
    
    #tedka udelam predikce
    rf_predikce_y = model_y.predict(X_testovaci)
    rf_predikce_x = model_x.predict(X_testovaci)
    
    #vratim vysledky v slovniku
    return {
        'skutecne_y': skutecne_y,
        'skutecne_x': skutecne_x,
        'zakladni_y': zakladni_y[testovaci_indexy],
        'zakladni_x': zakladni_x[testovaci_indexy],
        'rf_y': rf_predikce_y,
        'rf_x': rf_predikce_x,
        'model_y': model_y,
        'model_x': model_x
    }


def vypocitat_mse(predikce):
    """Vypočítá MSE pro skutečné vs. predikované hodnoty."""
    zakladni_mse = mean_squared_error(
        np.column_stack((predikce['skutecne_y'], predikce['skutecne_x'])),
        np.column_stack((predikce['zakladni_y'], predikce['zakladni_x']))
    )
    
    rf_mse = mean_squared_error(
        np.column_stack((predikce['skutecne_y'], predikce['skutecne_x'])),
        np.column_stack((predikce['rf_y'], predikce['rf_x']))
    )
    
    return zakladni_mse, rf_mse


def vypocitat_chyby(vysledky_bodu, id_bodu, testovaci_indexy):
    """Vypočítá chyby pro každý bod a každou metodu."""
    skutecne_hodnoty = pd.DataFrame(index=testovaci_indexy)
    zakladni_hodnoty = pd.DataFrame(index=testovaci_indexy)
    rf_hodnoty = pd.DataFrame(index=testovaci_indexy)
    zakladni_chyby = pd.DataFrame(index=testovaci_indexy)
    rf_chyby = pd.DataFrame(index=testovaci_indexy)
    
    for id_bod in id_bodu:
        prefix_bodu = f"kp{id_bod}"
        predikce = vysledky_bodu[id_bod]
        
        #ulozim hodnoty do dataframu
        skutecne_hodnoty[f'{prefix_bodu}_y'] = predikce['skutecne_y']
        skutecne_hodnoty[f'{prefix_bodu}_x'] = predikce['skutecne_x']
        
        zakladni_hodnoty[f'{prefix_bodu}_y'] = predikce['zakladni_y']
        zakladni_hodnoty[f'{prefix_bodu}_x'] = predikce['zakladni_x']
        
        rf_hodnoty[f'{prefix_bodu}_y'] = predikce['rf_y']
        rf_hodnoty[f'{prefix_bodu}_x'] = predikce['rf_x']
        
        #vypocitam chyby pro Y souradnice
        zakladni_chyby[f'{prefix_bodu}'] = np.abs(predikce['skutecne_y'] - predikce['zakladni_y'])
        rf_chyby[f'{prefix_bodu}'] = np.abs(predikce['skutecne_y'] - predikce['rf_y'])
    
    #vypocet celkovy mse, docela slozity
    def vypocet_celkove_mse(pravdive_df, predikce_df):
        pravdive_hodnoty = []
        predikovane_hodnoty = []
        
        for id_bod in id_bodu:
            prefix_bodu = f"kp{id_bod}"
            pravdive_hodnoty.append(pravdive_df[[f'{prefix_bodu}_y', f'{prefix_bodu}_x']].values)
            predikovane_hodnoty.append(predikce_df[[f'{prefix_bodu}_y', f'{prefix_bodu}_x']].values)
        
        pravdive_hodnoty = np.concatenate(pravdive_hodnoty, axis=1)
        predikovane_hodnoty = np.concatenate(predikovane_hodnoty, axis=1)
        
        return mean_squared_error(pravdive_hodnoty, predikovane_hodnoty)
    
    celkova_zakladni_mse = vypocet_celkove_mse(skutecne_hodnoty, zakladni_hodnoty)
    celkova_rf_mse = vypocet_celkove_mse(skutecne_hodnoty, rf_hodnoty)
    celkove_zlepseni = 100 * (celkova_zakladni_mse - celkova_rf_mse) / celkova_zakladni_mse
    
    print(f"\nCelkové výsledky:")
    print(f"MSE bodu s největší vahou: {celkova_zakladni_mse:.4f}")
    print(f"MSE náhodného lesa: {celkova_rf_mse:.4f}")
    print(f"Celkové zlepšení: {celkove_zlepseni:.2f}%")
    
    return skutecne_hodnoty, zakladni_hodnoty, rf_hodnoty, zakladni_chyby, rf_chyby


def ulozit_predikce(skutecne_hodnoty, zakladni_hodnoty, rf_hodnoty, zakladni_chyby, rf_chyby, id_bodu, vystupni_cesta):
    """Uloží predikce do CSV souboru."""
    print(f"Ukládání predikcí do {vystupni_cesta}...")
    vysledky_df = pd.DataFrame(index=skutecne_hodnoty.index)
    
    for id_bod in id_bodu:
        prefix_bodu = f"kp{id_bod}"
        
        #beru jen y souradnice a chyby pro prehlednost
        vysledky_df[f'skutecna_{prefix_bodu}_y'] = skutecne_hodnoty[f'{prefix_bodu}_y']
        vysledky_df[f'nejvetsi_vaha_{prefix_bodu}_y'] = zakladni_hodnoty[f'{prefix_bodu}_y']
        vysledky_df[f'predikce_{prefix_bodu}_y'] = rf_hodnoty[f'{prefix_bodu}_y']
        vysledky_df[f'chyba_nejvetsi_vaha_{prefix_bodu}'] = zakladni_chyby[f'{prefix_bodu}']
        vysledky_df[f'chyba_predikce_{prefix_bodu}'] = rf_chyby[f'{prefix_bodu}']
    
    #vytvorim adresar kdyztak
    os.makedirs(os.path.dirname(vystupni_cesta), exist_ok=True)
    vysledky_df.to_csv(vystupni_cesta)
    print(f"Predikce uloženy do {vystupni_cesta}")


def predikce_bodu(cesta_vstup="data/data-recovery.csv", cesta_vystup="data/predikce_bodu.csv"):
    """Hlavní funkce pro zpracování a predikci všech bodů."""
    cas_zacatek = time.time()
    
    #nactu data z csv
    df = nacist_data(cesta_vstup)
    
    #najdu vsechny body
    id_bodu = najit_body(df)
    
    #rozdelim data na trenink a test
    trenovaci_indexy, testovaci_indexy = rozdelit_data(df)
    
    #zpracuju kazdy bod zvlast
    print("Zpracování bodů...")
    vysledky_bodu = {}
    for id_bod in tqdm(id_bodu, desc="Zpracování bodů"):
        vysledky_bodu[id_bod] = zpracovat_bod(df, id_bod, trenovaci_indexy, testovaci_indexy)
    
    #spocitam chyby a tak
    skutecne_hodnoty, zakladni_hodnoty, rf_hodnoty, zakladni_chyby, rf_chyby = vypocitat_chyby(
        vysledky_bodu, id_bodu, testovaci_indexy
    )
    
    #ulozim vysledky do souboru
    ulozit_predikce(skutecne_hodnoty, zakladni_hodnoty, rf_hodnoty, zakladni_chyby, rf_chyby, id_bodu, cesta_vystup)
    
    cas_konec = time.time()
    print(f"Celková doba zpracování: {cas_konec - cas_zacatek:.2f} sekund")
    
    return vysledky_bodu


if __name__ == "__main__":
    predikce_bodu()