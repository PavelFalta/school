cesta = "data/data-recovery.csv"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import time
import re

# Load data - Načtení dat
df = pd.read_csv(cesta)
print("Velikost dat:", df.shape)

# Identify all keypoints - Identifikace všech bodů
vsechny_sloupce = df.columns
vzor_bodu = re.compile(r'target_kp(\d+)_[xy]')
shody_bodu = [vzor_bodu.match(sloupec) for sloupec in vsechny_sloupce]
id_bodu = sorted(list(set([int(shoda.group(1)) for shoda in shody_bodu if shoda])))

print(f"Nalezeno {len(id_bodu)} bodů: {id_bodu}")

# Split data - Rozdělení dat
indexy_radku = np.arange(len(df))
trenovaci_indexy, testovaci_indexy = train_test_split(indexy_radku, test_size=0.2, random_state=42)

# Set up results - Příprava výsledků
vysledky = []
skutecne_hodnoty = pd.DataFrame(index=testovaci_indexy)
zakladni_hodnoty = pd.DataFrame(index=testovaci_indexy)  # baseline
rf_hodnoty = pd.DataFrame(index=testovaci_indexy)       # random forest

# Process each keypoint - Zpracování každého bodu
cas_zacatek = time.time()

for id_bod in id_bodu:
    print(f"\n{'='*50}")
    print(f"Zpracování bodu {id_bod}")
    print(f"{'='*50}")
    
    prefix_bodu = f"kp{id_bod}"
    
    # Extract target columns - Získání cílových sloupců
    sloupec_y = f"target_{prefix_bodu}_y"
    sloupec_x = f"target_{prefix_bodu}_x"
    
    # Store ground truth - Uložení skutečných hodnot
    skutecne_hodnoty[f'{prefix_bodu}_y'] = df.loc[testovaci_indexy, sloupec_y].values
    skutecne_hodnoty[f'{prefix_bodu}_x'] = df.loc[testovaci_indexy, sloupec_x].values
    
    # Calculate baseline - Výpočet základních hodnot (baseline)
    vahy_sloupce = [f'pred_{prefix_bodu}_val{i}' for i in range(5)]
    indexy_nejvyssi_vahy = np.argmax(df.loc[:, vahy_sloupce].values, axis=1)
    
    zakladni_y = np.zeros(len(df))
    zakladni_x = np.zeros(len(df))
    
    for i in range(len(df)):
        idx = indexy_nejvyssi_vahy[i]
        zakladni_y[i] = df.iloc[i][f'pred_{prefix_bodu}_pos{idx}_y']
        zakladni_x[i] = df.iloc[i][f'pred_{prefix_bodu}_pos{idx}_x']
    
    # Store baseline - Uložení základních hodnot
    zakladni_hodnoty[f'{prefix_bodu}_y'] = zakladni_y[testovaci_indexy]
    zakladni_hodnoty[f'{prefix_bodu}_x'] = zakladni_x[testovaci_indexy]
    
    # Create features - Vytvoření příznaků
    priznaky_sloupce = []
    
    # Position predictions - Predikce pozic
    for i in range(5):
        priznaky_sloupce.extend([f'pred_{prefix_bodu}_pos{i}_y', f'pred_{prefix_bodu}_pos{i}_x'])
    
    # Weight values - Hodnoty vah
    priznaky_sloupce.extend([f'pred_{prefix_bodu}_val{i}' for i in range(5)])
    
    # Centroid and sigma - Centroid a sigma
    priznaky_sloupce.extend([
        f'pred_{prefix_bodu}_centroid_y', f'pred_{prefix_bodu}_centroid_x',
        f'pred_{prefix_bodu}_sigma_y', f'pred_{prefix_bodu}_sigma_x'
    ])
    
    # Extract and scale features - Extrakce a škálování příznaků
    X = df[priznaky_sloupce].values
    skaler = StandardScaler()
    X_skalovane = skaler.fit_transform(X)
    
    # Split into train and test - Rozdělení na trénovací a testovací sadu
    X_trenovaci = X_skalovane[trenovaci_indexy]
    X_testovaci = X_skalovane[testovaci_indexy]
    
    y_trenovaci_y = df.loc[trenovaci_indexy, sloupec_y].values
    y_trenovaci_x = df.loc[trenovaci_indexy, sloupec_x].values
    
    y_testovaci_y = df.loc[testovaci_indexy, sloupec_y].values
    y_testovaci_x = df.loc[testovaci_indexy, sloupec_x].values
    
    # Train Random Forest - Trénování náhodného lesa
    print(f"Trénování náhodného lesa pro souřadnici Y bodu {prefix_bodu}...")
    rf_y = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_y.fit(X_trenovaci, y_trenovaci_y)
    
    print(f"Trénování náhodného lesa pro souřadnici X bodu {prefix_bodu}...")
    rf_x = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_x.fit(X_trenovaci, y_trenovaci_x)
    
    # Make predictions - Vytvoření predikcí
    rf_predikce_y = rf_y.predict(X_testovaci)
    rf_predikce_x = rf_x.predict(X_testovaci)
    
    # Store Random Forest predictions - Uložení predikcí náhodného lesa
    rf_hodnoty[f'{prefix_bodu}_y'] = rf_predikce_y
    rf_hodnoty[f'{prefix_bodu}_x'] = rf_predikce_x
    
    # Calculate MSE - Výpočet střední kvadratické chyby
    zakladni_mse = mean_squared_error(
        np.column_stack((y_testovaci_y, y_testovaci_x)),
        np.column_stack((zakladni_y[testovaci_indexy], zakladni_x[testovaci_indexy]))
    )
    
    rf_mse = mean_squared_error(
        np.column_stack((y_testovaci_y, y_testovaci_x)),
        np.column_stack((rf_predikce_y, rf_predikce_x))
    )
    
    # Calculate improvement - Výpočet zlepšení
    zlepseni = 100 * (zakladni_mse - rf_mse) / zakladni_mse
    
    print(f"Základní MSE: {zakladni_mse:.4f}")
    print(f"RF MSE: {rf_mse:.4f}")
    print(f"Zlepšení: {zlepseni:.2f}%")
    
    # Show feature importance - Zobrazení důležitosti příznaků
    dulezitost_priznaku = pd.DataFrame({
        'Příznak': priznaky_sloupce,
        'Důležitost Y': rf_y.feature_importances_,
        'Důležitost X': rf_x.feature_importances_
    }).sort_values(by='Důležitost Y', ascending=False)
    
    print("\nTop 5 příznaků pro Y:")
    print(dulezitost_priznaku[['Příznak', 'Důležitost Y']].head(5))
    
    print("\nTop 5 příznaků pro X:")
    print(dulezitost_priznaku[['Příznak', 'Důležitost X']].sort_values(by='Důležitost X', ascending=False).head(5))
    
    # Store results - Uložení výsledků
    vysledky.append({
        'id_bod': id_bod,
        'zakladni_mse': zakladni_mse,
        'rf_mse': rf_mse,
        'zlepseni': zlepseni,
        'dulezitost_priznaku': dulezitost_priznaku
    })

celkovy_cas = time.time() - cas_zacatek
print(f"\nCelkový čas zpracování: {celkovy_cas:.2f} sekund")

# Calculate overall MSE - Výpočet celkové střední kvadratické chyby
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

print("\nCelkové výsledky:")
print(f"Základní MSE: {celkova_zakladni_mse:.4f}")
print(f"RF MSE: {celkova_rf_mse:.4f}")
print(f"Celkové zlepšení: {celkove_zlepseni:.2f}%")

# Create summary DataFrame - Vytvoření souhrnné tabulky
vysledky_df = pd.DataFrame([{
    'Bod': r['id_bod'],
    'Základní MSE': r['zakladni_mse'],
    'RF MSE': r['rf_mse'],
    'Zlepšení (%)': r['zlepseni']
} for r in vysledky])

print("\nVýsledky podle bodů:")
print(vysledky_df)

# Visualize results - Vizualizace výsledků
plt.figure(figsize=(15, 6))

# Plot MSE comparison - Graf srovnání MSE
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

# Plot improvement percentage - Graf procenta zlepšení
plt.subplot(1, 2, 2)
plt.bar(vysledky_df['Bod'].astype(str), vysledky_df['Zlepšení (%)'], color='blue')
plt.xlabel('ID bodu')
plt.ylabel('Zlepšení (%)')
plt.title('Zlepšení náhodného lesa oproti základnímu modelu')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Visualize sample rows - Vizualizace ukázkových řádků
plt.figure(figsize=(15, 10))

# Select random sample rows - Výběr náhodných ukázkových řádků
ukazkove_indexy = np.random.choice(testovaci_indexy, min(6, len(testovaci_indexy)), replace=False)

for i, idx in enumerate(ukazkove_indexy, 1):
    plt.subplot(2, 3, i)
    
    # Plot each keypoint - Vykreslení každého bodu
    for id_bod in id_bodu:
        prefix_bodu = f"kp{id_bod}"
        
        # Get positions - Získání pozic
        skutecna_y = skutecne_hodnoty.loc[idx, f'{prefix_bodu}_y']
        skutecna_x = skutecne_hodnoty.loc[idx, f'{prefix_bodu}_x']
        
        zakladni_y = zakladni_hodnoty.loc[idx, f'{prefix_bodu}_y']
        zakladni_x = zakladni_hodnoty.loc[idx, f'{prefix_bodu}_x']
        
        rf_y = rf_hodnoty.loc[idx, f'{prefix_bodu}_y']
        rf_x = rf_hodnoty.loc[idx, f'{prefix_bodu}_x']
        
        # Plot positions - Vykreslení pozic
        plt.scatter(skutecna_x, skutecna_y, color='blue', marker='o', s=80, label='Skutečná' if id_bod == id_bodu[0] else "")
        plt.scatter(zakladni_x, zakladni_y, color='red', marker='s', s=50, label='Základní' if id_bod == id_bodu[0] else "")
        plt.scatter(rf_x, rf_y, color='green', marker='^', s=50, label='Náhodný les' if id_bod == id_bodu[0] else "")
        
        # Draw lines - Vykreslení čar
        plt.plot([skutecna_x, zakladni_x], [skutecna_y, zakladni_y], 'r-', alpha=0.3)
        plt.plot([skutecna_x, rf_x], [skutecna_y, rf_y], 'g-', alpha=0.3)
        
        # Annotate keypoint id - Popis ID bodu
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
plt.show()

# Save predictions to CSV - Uložení predikcí do CSV
vysledky_df = pd.DataFrame(index=testovaci_indexy)

# Add predictions to result dataframe - Přidání predikcí do výsledné tabulky
for id_bod in id_bodu:
    prefix_bodu = f"kp{id_bod}"
    
    # True values - Skutečné hodnoty
    vysledky_df[f'skutecna_{prefix_bodu}_y'] = skutecne_hodnoty[f'{prefix_bodu}_y']
    vysledky_df[f'skutecna_{prefix_bodu}_x'] = skutecne_hodnoty[f'{prefix_bodu}_x']
    
    # Baseline predictions - Základní predikce
    vysledky_df[f'zakladni_{prefix_bodu}_y'] = zakladni_hodnoty[f'{prefix_bodu}_y']
    vysledky_df[f'zakladni_{prefix_bodu}_x'] = zakladni_hodnoty[f'{prefix_bodu}_x']
    
    # Random Forest predictions - Predikce náhodného lesa
    vysledky_df[f'rf_{prefix_bodu}_y'] = rf_hodnoty[f'{prefix_bodu}_y']
    vysledky_df[f'rf_{prefix_bodu}_x'] = rf_hodnoty[f'{prefix_bodu}_x']

# Save to CSV - Uložení do CSV
vystupni_cesta = "data/predikce_bodu.csv"
vysledky_df.to_csv(vystupni_cesta)
print(f"\nPredikce uloženy do {vystupni_cesta}")

# Display sample predictions - Zobrazení ukázkových predikcí
print("\nUkázkové predikce:")
print(vysledky_df.head())