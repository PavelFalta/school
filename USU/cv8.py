import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
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


def vytvorit_grafy_pro_latex(vysledky_bodu, id_bodu, testovaci_indexy, adresar_grafu="grafy"):
    """Vytváří grafy pro LaTeX zprávu."""
    # Vytvoření adresáře pro grafy
    os.makedirs(adresar_grafu, exist_ok=True)
    
    # Připrava dat pro grafy
    data_grafu = []
    for id_bod in id_bodu:
        predikce = vysledky_bodu[id_bod]
        zakladni_mse, rf_mse = vypocitat_mse(predikce)
        zlepseni = 100 * (zakladni_mse - rf_mse) / zakladni_mse
        data_grafu.append({
            'Bod': id_bod,
            'MSE s největší vahou': zakladni_mse,
            'MSE náhodného lesa': rf_mse,
            'Zlepšení (%)': zlepseni
        })
    
    vysledky_df = pd.DataFrame(data_grafu)
    
    # Výpočet celkového MSE
    celkova_zakladni_mse, celkova_rf_mse, skutecne_hodnoty, zakladni_hodnoty, rf_hodnoty = vypocitat_celkove_mse(
        vysledky_bodu, id_bodu, testovaci_indexy
    )
    celkove_zlepseni = 100 * (celkova_zakladni_mse - celkova_rf_mse) / celkova_zakladni_mse
    
    print(f"\nCelkové výsledky:")
    print(f"MSE bodu s největší vahou: {celkova_zakladni_mse:.4f}")
    print(f"MSE náhodného lesa: {celkova_rf_mse:.4f}")
    print(f"Celkové zlepšení: {celkove_zlepseni:.2f}%")
    
    # Nastavení stylu pro publikační kvalitu
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'text.usetex': False,  # Nastavte na True, pokud máte nainstalovaný LaTeX
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'figure.autolayout': True,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.5,
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05
    })
    
    # 1. Graf MSE pro jednotlivé body
    plt.figure(figsize=(10, 6))
    index = np.arange(len(id_bodu))
    sirka = 0.35
    
    plt.bar(index - sirka/2, vysledky_df['MSE s největší vahou'], sirka, 
            label='Bod s největší vahou', color='#FF7F50', alpha=0.8)
    plt.bar(index + sirka/2, vysledky_df['MSE náhodného lesa'], sirka, 
            label='Náhodný les', color='#4682B4', alpha=0.8)
    
    # Přidání hodnot do sloupců
    for i, v in enumerate(vysledky_df['MSE s největší vahou']):
        plt.text(i - sirka/2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    for i, v in enumerate(vysledky_df['MSE náhodného lesa']):
        plt.text(i + sirka/2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('ID klíčového bodu')
    plt.ylabel('Střední kvadratická chyba (MSE)')
    plt.title('Porovnání MSE podle metodiky predikce')
    plt.xticks(index, vysledky_df['Bod'].astype(str))
    plt.legend(loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{adresar_grafu}/mse_porovnani.pdf")
    plt.savefig(f"{adresar_grafu}/mse_porovnani.png")
    plt.close()
    
    # 2. Graf procentuálního zlepšení
    plt.figure(figsize=(10, 6))
    plt.bar(vysledky_df['Bod'].astype(str), vysledky_df['Zlepšení (%)'], color='#6495ED')
    
    # Přidání hodnot do sloupců
    for i, v in enumerate(vysledky_df['Zlepšení (%)']):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')
    
    plt.axhline(y=celkove_zlepseni, color='r', linestyle='--', label=f'Průměrné zlepšení: {celkove_zlepseni:.1f}%')
    plt.xlabel('ID klíčového bodu')
    plt.ylabel('Zlepšení (%)')
    plt.title('Procentuální zlepšení náhodného lesa oproti bodu s největší vahou')
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.savefig(f"{adresar_grafu}/zlepseni_procenta.pdf")
    plt.savefig(f"{adresar_grafu}/zlepseni_procenta.png")
    plt.close()
    
    # 3. Celkové srovnání MSE
    plt.figure(figsize=(8, 6))
    metody = ['Bod s největší vahou', 'Náhodný les']
    celkove_mse = [celkova_zakladni_mse, celkova_rf_mse]
    barvy = ['#FF7F50', '#4682B4']
    
    sloupce = plt.bar(metody, celkove_mse, color=barvy, width=0.6)
    
    # Přidání hodnot do sloupců
    for sloupec in sloupce:
        height = sloupec.get_height()
        plt.text(sloupec.get_x() + sloupec.get_width()/2, height + 0.01,
                f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Celková MSE (všechny body)')
    plt.title('Celkové porovnání přesnosti metod predikce')
    plt.grid(axis='y', linestyle='--')
    
    # Přidání anotace zlepšení
    plt.annotate(f'Zlepšení: {celkove_zlepseni:.1f}%', 
                xy=(1, celkova_rf_mse), 
                xytext=(1.2, celkova_rf_mse + (celkova_zakladni_mse - celkova_rf_mse)/2),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{adresar_grafu}/celkove_porovnani.pdf")
    plt.savefig(f"{adresar_grafu}/celkove_porovnani.png")
    plt.close()
    
    # 4. Ukázkové predikce na náhodných vzorcích
    ukazkove_indexy = np.random.choice(testovaci_indexy, min(3, len(testovaci_indexy)), replace=False)
    
    for idx in ukazkove_indexy:
        plt.figure(figsize=(8, 8))
        
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
            
            # Vykreslení pozic s většími značkami pro lepší viditelnost
            plt.scatter(skutecna_x, skutecna_y, color='blue', marker='o', s=100, 
                       label='Skutečná poloha' if id_bod == id_bodu[0] else "")
            plt.scatter(zakladni_x, zakladni_y, color='#FF7F50', marker='s', s=80, 
                       label='Bod s největší vahou' if id_bod == id_bodu[0] else "")
            plt.scatter(rf_x, rf_y, color='#4682B4', marker='^', s=80, 
                       label='Náhodný les' if id_bod == id_bodu[0] else "")
            
            # Vykreslení čar s vyšší průhledností pro lepší čitelnost
            plt.plot([skutecna_x, zakladni_x], [skutecna_y, zakladni_y], 
                    color='#FF7F50', linestyle='-', alpha=0.4)
            plt.plot([skutecna_x, rf_x], [skutecna_y, rf_y], 
                    color='#4682B4', linestyle='-', alpha=0.4)
            
            # Popis ID bodu s větším písmem
            plt.annotate(str(id_bod), (skutecna_x, skutecna_y), fontsize=10, ha='right', 
                        fontweight='bold', xytext=(-5, 0), textcoords='offset points')
        
        plt.title(f'Ukázka predikce bodů pro řádek {idx}')
        plt.xlabel('Souřadnice X')
        plt.ylabel('Souřadnice Y')
        plt.legend(loc='best', frameon=True, framealpha=0.9)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(f"{adresar_grafu}/ukazka_predikce_{idx}.pdf")
        plt.savefig(f"{adresar_grafu}/ukazka_predikce_{idx}.png")
        plt.close()
    
    # 5. Graf rozložení chyb
    plt.figure(figsize=(12, 6))
    
    # Výpočet chyb pro každý bod
    chyby_zakladni = []
    chyby_rf = []
    
    for id_bod in id_bodu:
        prefix_bodu = f"kp{id_bod}"
        for idx in testovaci_indexy:
            # Získání pozic
            skutecna_y = skutecne_hodnoty.loc[idx, f'{prefix_bodu}_y']
            skutecna_x = skutecne_hodnoty.loc[idx, f'{prefix_bodu}_x']
            
            zakladni_y = zakladni_hodnoty.loc[idx, f'{prefix_bodu}_y']
            zakladni_x = zakladni_hodnoty.loc[idx, f'{prefix_bodu}_x']
            
            rf_y = rf_hodnoty.loc[idx, f'{prefix_bodu}_y']
            rf_x = rf_hodnoty.loc[idx, f'{prefix_bodu}_x']
            
            # Výpočet Euklidovské vzdálenosti
            chyba_zakladni = np.sqrt((skutecna_y - zakladni_y)**2 + (skutecna_x - zakladni_x)**2)
            chyba_rf = np.sqrt((skutecna_y - rf_y)**2 + (skutecna_x - rf_x)**2)
            
            chyby_zakladni.append(chyba_zakladni)
            chyby_rf.append(chyba_rf)
    
    # Vytvoření histogramů
    plt.subplot(1, 2, 1)
    plt.hist(chyby_zakladni, bins=30, alpha=0.7, color='#FF7F50', edgecolor='black', linewidth=0.8)
    plt.axvline(np.mean(chyby_zakladni), color='r', linestyle='--', 
                label=f'Průměr: {np.mean(chyby_zakladni):.2f}')
    plt.xlabel('Chyba (Euklidovská vzdálenost)')
    plt.ylabel('Počet výskytů')
    plt.title('Distribuce chyb: Bod s největší vahou')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(chyby_rf, bins=30, alpha=0.7, color='#4682B4', edgecolor='black', linewidth=0.8)
    plt.axvline(np.mean(chyby_rf), color='r', linestyle='--', 
                label=f'Průměr: {np.mean(chyby_rf):.2f}')
    plt.xlabel('Chyba (Euklidovská vzdálenost)')
    plt.ylabel('Počet výskytů')
    plt.title('Distribuce chyb: Náhodný les')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{adresar_grafu}/distribuce_chyb.pdf")
    plt.savefig(f"{adresar_grafu}/distribuce_chyb.png")
    plt.close()
    
    return skutecne_hodnoty, zakladni_hodnoty, rf_hodnoty


def ulozit_predikce(skutecne_hodnoty, zakladni_hodnoty, rf_hodnoty, id_bodu, vystupni_cesta):
    """Uloží predikce do CSV souboru."""
    print(f"Ukládání predikcí do {vystupni_cesta}...")
    vysledky_df = pd.DataFrame(index=skutecne_hodnoty.index)
    
    for id_bod in id_bodu:
        prefix_bodu = f"kp{id_bod}"
        
        # Skutečné hodnoty
        vysledky_df[f'skutecna_{prefix_bodu}_y'] = skutecne_hodnoty[f'{prefix_bodu}_y']
        vysledky_df[f'skutecna_{prefix_bodu}_x'] = skutecne_hodnoty[f'{prefix_bodu}_x']
        
        # Základní predikce
        vysledky_df[f'nejvetsi_vaha_{prefix_bodu}_y'] = zakladni_hodnoty[f'{prefix_bodu}_y']
        vysledky_df[f'nejvetsi_vaha_{prefix_bodu}_x'] = zakladni_hodnoty[f'{prefix_bodu}_x']
        
        # Predikce náhodného lesa
        vysledky_df[f'rf_{prefix_bodu}_y'] = rf_hodnoty[f'{prefix_bodu}_y']
        vysledky_df[f'rf_{prefix_bodu}_x'] = rf_hodnoty[f'{prefix_bodu}_x']
    
    # Ujistíme se, že existuje adresář pro výstup
    os.makedirs(os.path.dirname(vystupni_cesta), exist_ok=True)
    vysledky_df.to_csv(vystupni_cesta)
    print(f"Predikce uloženy do {vystupni_cesta}")


def predikce_bodu(cesta_vstup="data/data-recovery.csv", cesta_vystup="data/predikce_bodu.csv"):
    """Hlavní funkce pro zpracování a predikci všech bodů."""
    cas_zacatek = time.time()
    
    # Načtení dat
    df = nacist_data(cesta_vstup)
    
    # Nalezení ID bodů
    id_bodu = najit_body(df)
    
    # Rozdělení dat
    trenovaci_indexy, testovaci_indexy = rozdelit_data(df)
    
    # Zpracování každého bodu
    print("Zpracování bodů...")
    vysledky_bodu = {}
    for id_bod in tqdm(id_bodu, desc="Zpracování bodů"):
        vysledky_bodu[id_bod] = zpracovat_bod(df, id_bod, trenovaci_indexy, testovaci_indexy)
    
    # Vytvoření grafů pro LaTeX zprávu
    print("Vytváření grafů pro LaTeX zprávu...")
    skutecne_hodnoty, zakladni_hodnoty, rf_hodnoty = vytvorit_grafy_pro_latex(
        vysledky_bodu, id_bodu, testovaci_indexy
    )
    
    # Uložení predikcí
    ulozit_predikce(skutecne_hodnoty, zakladni_hodnoty, rf_hodnoty, id_bodu, cesta_vystup)
    
    cas_konec = time.time()
    print(f"Celková doba zpracování: {cas_konec - cas_zacatek:.2f} sekund")
    
    return vysledky_bodu


if __name__ == "__main__":
    predikce_bodu()