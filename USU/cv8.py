import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import re
import os
from tqdm import tqdm
from warnings import filterwarnings
import optuna
import json
import joblib
from sklearn.metrics import mean_squared_error

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


def rozdel_data(df, test_velikost=0.2, random_state=42):
    radky_indexy = np.arange(len(df))
    train_idx, test_idx = train_test_split(radky_indexy, test_size=test_velikost, random_state=random_state)
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


def trenuj_model(X_train, y_train, params):
    model = RandomForestRegressor(random_state=42, **params)
    model.fit(X_train, y_train)
    return model


def zpracuj_bod(df, cislo_bodu, train_idx, test_idx, params_dir):
    prefix = f"kp{cislo_bodu}"
    
    # Define path for saving/loading best hyperparameters
    param_path = os.path.join(params_dir, f"kp{cislo_bodu}_best_params.json")
    os.makedirs(params_dir, exist_ok=True) # Ensure directory exists
    
    #najdu cilovy sloupecky
    y_sloupec = f"target_{prefix}_y"
    x_sloupec = f"target_{prefix}_x"
    
    #ulozim si skutecny hodnoty
    y_pravda = df.loc[test_idx, y_sloupec].values
    x_pravda = df.loc[test_idx, x_sloupec].values
    
    #tady ziskam zakladni predikce
    y_zaklad, x_zaklad = udel_zakladni_predikce(df, cislo_bodu)
    
    #vytvori priznaky
    sloupce_priznaku = udel_priznaky(df, cislo_bodu)
    X = df[sloupce_priznaku].values

    # Rozdelim data pred skalovanim
    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]

    # Vytvorim a natrenuju skalovac na trenovacich datech
    skalovac = StandardScaler()
    skalovac.fit(X_train_raw)

    # Oskaluju trenovaci a testovaci data
    X_train = skalovac.transform(X_train_raw)
    X_test = skalovac.transform(X_test_raw)

    y_train_y = df.loc[train_idx, y_sloupec].values
    y_train_x = df.loc[train_idx, x_sloupec].values
    
    best_params_y = None
    best_params_x = None

    # --- Optuna Integration --- 
    try:
        # Try loading existing best parameters
        with open(param_path, 'r') as f:
            saved_params = json.load(f)
            best_params_y = saved_params['y']
            best_params_x = saved_params['x']
        print(f"Načteny nejlepší parametry pro bod {cislo_bodu} z {param_path}")
    except FileNotFoundError:
        print(f"Nejlepší parametry pro bod {cislo_bodu} nenalezeny. Spouštění Optuna optimalizace...")
        
        # Split training data further for hyperparameter tuning validation
        X_train_opt, X_val_opt, y_train_y_opt, y_val_y_opt = train_test_split(
            X_train, y_train_y, test_size=0.25, random_state=42) # Use 25% for validation
        _, _, y_train_x_opt, y_val_x_opt = train_test_split(
            X_train, y_train_x, test_size=0.25, random_state=42) # Need same split for X features

        # --- Objective Function for Y --- 
        def objective_y(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 10, 100, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error'])
            }
            model = RandomForestRegressor(random_state=42, **params)
            model.fit(X_train_opt, y_train_y_opt)
            preds = model.predict(X_val_opt)
            mse = mean_squared_error(y_val_y_opt, preds)
            return mse

        # --- Objective Function for X --- 
        def objective_x(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
                'max_depth': trial.suggest_int('max_depth', 10, 100, log=True),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5, 0.7, 1.0]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'criterion': trial.suggest_categorical('criterion', ['squared_error', 'absolute_error'])
            }
            model = RandomForestRegressor(random_state=42, **params)
            model.fit(X_train_opt, y_train_x_opt)
            preds = model.predict(X_val_opt)
            mse = mean_squared_error(y_val_x_opt, preds)
            return mse
        
        # --- Run Optuna Study for Y --- 
        study_y = optuna.create_study(direction='minimize')
        study_y.optimize(objective_y, n_trials=150, show_progress_bar=False)
        best_params_y = study_y.best_params
        print(f"Optimalizace Y pro bod {cislo_bodu} dokončena. Nejlepší parametry: {best_params_y}")

        # --- Run Optuna Study for X --- 
        study_x = optuna.create_study(direction='minimize')
        study_x.optimize(objective_x, n_trials=150, show_progress_bar=False)
        best_params_x = study_x.best_params
        print(f"Optimalizace X pro bod {cislo_bodu} dokončena. Nejlepší parametry: {best_params_x}")

        # Save the best parameters found
        with open(param_path, 'w') as f:
            json.dump({'y': best_params_y, 'x': best_params_x}, f, indent=4)
        print(f"Nejlepší parametry uloženy do {param_path}")
    
    # --- Train Final Models with Best Parameters --- 
    print(f"Trénování finálních modelů pro bod {cislo_bodu} s nejlepšími parametry...")
    #trenuju modely
    y_model = trenuj_model(X_train, y_train_y, best_params_y)
    x_model = trenuj_model(X_train, y_train_x, best_params_x)
    
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

def vypocitej_pro_bod(vysledky, cisla_bodu, test_idx):
    pravdive_hodnoty = pd.DataFrame(index=test_idx)
    zakladni_hodnoty = pd.DataFrame(index=test_idx)
    rf_hodnoty = pd.DataFrame(index=test_idx)
    
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
    
    
    return pravdive_hodnoty, zakladni_hodnoty, rf_hodnoty


def uloz_predikce(pravdive_hodnoty, zakladni_hodnoty, rf_hodnoty, cisla_bodu, cesta_vystup, ran_i):
    print(f"Ukládání predikcí do {cesta_vystup}...")
    vysledky_df = pd.DataFrame(index=pravdive_hodnoty.index)
    
    for cislo in cisla_bodu:
        prefix = f"kp{cislo}"
        
        #beru jen souradnice a chyby pro prehlednost
        vysledky_df[f'skutecna_{prefix}_y'] = pravdive_hodnoty[f'{prefix}_y']
        vysledky_df[f'skutecna_{prefix}_x'] = pravdive_hodnoty[f'{prefix}_x']
        
        vysledky_df[f'nejvetsi_vaha_{prefix}_y'] = zakladni_hodnoty[f'{prefix}_y']
        vysledky_df[f'nejvetsi_vaha_{prefix}_x'] = zakladni_hodnoty[f'{prefix}_x']
        
        vysledky_df[f'predikce_{prefix}_y'] = rf_hodnoty[f'{prefix}_y']
        vysledky_df[f'predikce_{prefix}_x'] = rf_hodnoty[f'{prefix}_x']
        
    
    #vytvorim adresar kdyztak
    os.makedirs(os.path.dirname(cesta_vystup), exist_ok=True)
    vysledky_df.to_csv(cesta_vystup + f"_ran{ran_i}.csv", index=False)
    print(f"Predikce uloženy do {cesta_vystup}")


def predikuj_body(cesta_vstup="data/data-recovery.csv", cesta_vystup="data/predikce_pavel/predikce_bodu"):
    #nactu data z csv
    df = nacti_data(cesta_vstup)
    
    #najdu vsechny body
    cisla_bodu = najdi_body(df)
    
    #rozdelim data na trenink a test
    for ran_i in range(10):
        train_idx, test_idx = rozdel_data(df, random_state=ran_i)
        
        #zpracuju kazdy bod zvlast
        print("Zpracování bodů...")
        vysledky = {}
        params_dir = "data/best_params" # Define parameter directory
        for cislo in tqdm(cisla_bodu, desc="Zpracování bodů"):
            vysledky[cislo] = zpracuj_bod(df, cislo, train_idx, test_idx, params_dir=params_dir) # Pass params_dir
        
        #spocitam chyby a tak
        pravdive_hodnoty, zakladni_hodnoty, rf_hodnoty = vypocitej_pro_bod(
            vysledky, cisla_bodu, test_idx
        )
        
        #ulozim vysledky do souboru
        uloz_predikce(pravdive_hodnoty, zakladni_hodnoty, rf_hodnoty, cisla_bodu, cesta_vystup, ran_i)
        
    return vysledky


if __name__ == "__main__":
    predikuj_body()