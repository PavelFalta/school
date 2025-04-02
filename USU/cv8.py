import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import re
import os


def nacist_data(cesta):
    """Načte data z CSV souboru."""
    return pd.read_csv(cesta)


def najit_body(df):
    """Najde všechny klíčové body v datasetu pomocí regex."""
    vsechny_sloupce = df.columns
    vzor_bodu = re.compile(r'target_kp(\d+)_[xy]')
    shody_bodu = [vzor_bodu.match(sloupec) for sloupec in vsechny_sloupce]
    return sorted(list(set([int(shoda.group(1)) for shoda in shody_bodu if shoda])))


def rozdelit_data(df, test_velikost=0.2):
    """Rozdělí data na trénovací a testovací sadu."""
    indexy_radku = np.arange(len(df))
    return train_test_split(indexy_radku, test_size=test_velikost, random_state=42)


def vytvorit_zakladni_predikce(df, id_bod):
    """Vytvoří základní predikce na základě nejvyšší váhy."""
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
    
    # Predikce pozic
    for i in range(5):
        priznaky_sloupce.extend([f'pred_{prefix_bodu}_pos{i}_y', f'pred_{prefix_bodu}_pos{i}_x'])
    
    # Hodnoty vah
    priznaky_sloupce.extend([f'pred_{prefix_bodu}_val{i}' for i in range(5)])
    
    # Centroid a sigma
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
    
    # Získání cílových sloupců
    sloupec_y = f"target_{prefix_bodu}_y"
    sloupec_x = f"target_{prefix_bodu}_x"
    
    # Uložení skutečných hodnot
    skutecne_y = df.loc[testovaci_indexy, sloupec_y].values
    skutecne_x = df.loc[testovaci_indexy, sloupec_x].values
    
    # Získání základních predikcí
    zakladni_y, zakladni_x = vytvorit_zakladni_predikce(df, id_bod)
    
    # Vytvoření a škálování příznaků
    priznaky_sloupce = vytvorit_priznaky(df, id_bod)
    X = df[priznaky_sloupce].values
    skaler = StandardScaler()
    X_skalovane = skaler.fit_transform(X)
    
    X_trenovaci = X_skalovane[trenovaci_indexy]
    X_testovaci = X_skalovane[testovaci_indexy]
    
    y_trenovaci_y = df.loc[trenovaci_indexy, sloupec_y].values
    y_trenovaci_x = df.loc[trenovaci_indexy, sloupec_x].values
    
    # Trénování modelů
    model_y = trenovat_model(X_trenovaci, y_trenovaci_y)
    model_x = trenovat_model(X_trenovaci, y_trenovaci_x)
    
    # Vytvoření predikcí
    rf_predikce_y = model_y.predict(X_testovaci)
    rf_predikce_x = model_x.predict(X_testovaci)
    
    # Sestavení výstupu
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


def vypocitat_celkove_mse(vysledky_bodu, id_bodu, testovaci_indexy):
    """Vypočítá celkové MSE napříč všemi body."""
    skutecne_hodnoty = pd.DataFrame(index=testovaci_indexy)
    zakladni_hodnoty = pd.DataFrame(index=testovaci_indexy)
    rf_hodnoty = pd.DataFrame(index=testovaci_indexy)
    
    for id_bod in id_bodu:
        prefix_bodu = f"kp{id_bod}"
        predikce = vysledky_bodu[id_bod]
        
        skutecne_hodnoty[f'{prefix_bodu}_y'] = predikce['skutecne_y']
        skutecne_hodnoty[f'{prefix_bodu}_x'] = predikce['skutecne_x']
        
        zakladni_hodnoty[f'{prefix_bodu}_y'] = predikce['zakladni_y']
        zakladni_hodnoty[f'{prefix_bodu}_x'] = predikce['zakladni_x']
        
        rf_hodnoty[f'{prefix_bodu}_y'] = predikce['rf_y']
        rf_hodnoty[f'{prefix_bodu}_x'] = predikce['rf_x']
    
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
    
    return celkova_zakladni_mse, celkova_rf_mse, skutecne_hodnoty, zakladni_hodnoty, rf_hodnoty


def zobrazit_vysledky(vysledky_bodu, id_bodu, testovaci_indexy):
    """Vytváří a zobrazuje grafy výsledků."""
    # Připrava dat pro grafy
    data_grafu = []
    for id_bod in id_bodu:
        predikce = vysledky_bodu[id_bod]
        zakladni_mse, rf_mse = vypocitat_mse(predikce)
        data_grafu.append({
            'Bod': id_bod,
            'Základní MSE': zakladni_mse,
            'RF MSE': rf_mse,
            'Zlepšení (%)': 100 * (zakladni_mse - rf_mse) / zakladni_mse
        })
    
    vysledky_df = pd.DataFrame(data_grafu)
    
    # Výpočet celkového MSE
    celkova_zakladni_mse, celkova_rf_mse, skutecne_hodnoty, zakladni_hodnoty, rf_hodnoty = vypocitat_celkove_mse(
        vysledky_bodu, id_bodu, testovaci_indexy
    )
    
    # Graf srovnání MSE a zlepšení
    plt.figure(figsize=(15, 6))
    
    # Graf MSE
    plt.subplot(1, 2, 1)
    index = np.arange(len(id_bodu))
    sirka = 0.35
    
    plt.bar(index - sirka/2, vysledky_df['Základní MSE'], sirka, label='Základní model', color='red', alpha=0.7)
    plt.bar(index + sirka/2, vysledky_df['RF MSE'], sirka, label='Náhodný les', color='green', alpha=0.7)
    
    plt.xlabel('ID bodu')
    plt.ylabel('MSE')
    plt.title('Střední kvadratická chyba podle bodu')
    plt.xticks(index, vysledky_df['Bod'].astype(str))
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Graf zlepšení
    plt.subplot(1, 2, 2)
    plt.bar(vysledky_df['Bod'].astype(str), vysledky_df['Zlepšení (%)'], color='blue')
    plt.xlabel('ID bodu')
    plt.ylabel('Zlepšení (%)')
    plt.title('Zlepšení náhodného lesa oproti základnímu modelu')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("vysledky_mse.png")
    plt.close()
    
    # Zobrazení ukázkových bodů
    plt.figure(figsize=(15, 10))
    
    # Výběr náhodných ukázkových řádků
    ukazkove_indexy = np.random.choice(testovaci_indexy, min(6, len(testovaci_indexy)), replace=False)
    
    for i, idx in enumerate(ukazkove_indexy, 1):
        plt.subplot(2, 3, i)
        
        # Vykreslení každého bodu
        for id_bod in id_bodu:
            prefix_bodu = f"kp{id_bod}"
            
            # Získání pozic
            skutecna_y = skutecne_hodnoty.loc[idx, f'{prefix_bodu}_y']
            skutecna_x = skutecne_hodnoty.loc[idx, f'{prefix_bodu}_x']
            
            zakladni_y = zakladni_hodnoty.loc[idx, f'{prefix_bodu}_y']
            zakladni_x = zakladni_hodnoty.loc[idx, f'{prefix_bodu}_x']
            
            rf_y = rf_hodnoty.loc[idx, f'{prefix_bodu}_y']
            rf_x = rf_hodnoty.loc[idx, f'{prefix_bodu}_x']
            
            # Vykreslení pozic
            plt.scatter(skutecna_x, skutecna_y, color='blue', marker='o', s=80, 
                       label='Skutečná' if id_bod == id_bodu[0] else "")
            plt.scatter(zakladni_x, zakladni_y, color='red', marker='s', s=50, 
                       label='Základní' if id_bod == id_bodu[0] else "")
            plt.scatter(rf_x, rf_y, color='green', marker='^', s=50, 
                       label='Náhodný les' if id_bod == id_bodu[0] else "")
            
            # Vykreslení čar
            plt.plot([skutecna_x, zakladni_x], [skutecna_y, zakladni_y], 'r-', alpha=0.3)
            plt.plot([skutecna_x, rf_x], [skutecna_y, rf_y], 'g-', alpha=0.3)
            
            # Popis ID bodu
            plt.annotate(str(id_bod), (skutecna_x, skutecna_y), fontsize=8, ha='right')
        
        plt.title(f'Řádek {idx}')
        plt.xlabel('Souřadnice X')
        plt.ylabel('Souřadnice Y')
        if i == 1:
            plt.legend()
        plt.grid(True)
        plt.axis('equal')
    
    plt.suptitle('Skutečné vs. predikované pozice bodů', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("ukazkove_predikce.png")
    plt.close()
    
    return skutecne_hodnoty, zakladni_hodnoty, rf_hodnoty


def ulozit_predikce(skutecne_hodnoty, zakladni_hodnoty, rf_hodnoty, id_bodu, vystupni_cesta):
    """Uloží predikce do CSV souboru."""
    vysledky_df = pd.DataFrame(index=skutecne_hodnoty.index)
    
    for id_bod in id_bodu:
        prefix_bodu = f"kp{id_bod}"
        
        # Skutečné hodnoty
        vysledky_df[f'skutecna_{prefix_bodu}_y'] = skutecne_hodnoty[f'{prefix_bodu}_y']
        vysledky_df[f'skutecna_{prefix_bodu}_x'] = skutecne_hodnoty[f'{prefix_bodu}_x']
        
        # Základní predikce
        vysledky_df[f'zakladni_{prefix_bodu}_y'] = zakladni_hodnoty[f'{prefix_bodu}_y']
        vysledky_df[f'zakladni_{prefix_bodu}_x'] = zakladni_hodnoty[f'{prefix_bodu}_x']
        
        # Predikce náhodného lesa
        vysledky_df[f'rf_{prefix_bodu}_y'] = rf_hodnoty[f'{prefix_bodu}_y']
        vysledky_df[f'rf_{prefix_bodu}_x'] = rf_hodnoty[f'{prefix_bodu}_x']
    
    # Ujistíme se, že existuje adresář pro výstup
    os.makedirs(os.path.dirname(vystupni_cesta), exist_ok=True)
    vysledky_df.to_csv(vystupni_cesta)


def predikce_bodu(cesta_vstup="data/data-recovery.csv", cesta_vystup="data/predikce_bodu.csv"):
    """Hlavní funkce pro zpracování a predikci všech bodů."""
    # Načtení dat
    df = nacist_data(cesta_vstup)
    
    # Nalezení ID bodů
    id_bodu = najit_body(df)
    
    # Rozdělení dat
    trenovaci_indexy, testovaci_indexy = rozdelit_data(df)
    
    # Zpracování každého bodu
    vysledky_bodu = {}
    for id_bod in id_bodu:
        vysledky_bodu[id_bod] = zpracovat_bod(df, id_bod, trenovaci_indexy, testovaci_indexy)
    
    # Zobrazení výsledků
    skutecne_hodnoty, zakladni_hodnoty, rf_hodnoty = zobrazit_vysledky(vysledky_bodu, id_bodu, testovaci_indexy)
    
    # Uložení predikcí
    ulozit_predikce(skutecne_hodnoty, zakladni_hodnoty, rf_hodnoty, id_bodu, cesta_vystup)
    
    return vysledky_bodu


if __name__ == "__main__":
    predikce_bodu()