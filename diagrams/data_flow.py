import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrow, FancyBboxPatch, Circle

# nastaveni velikosti a stylu
plt.figure(figsize=(14, 12))
plt.style.use('ggplot')

ax = plt.gca()
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# barvy
barvy = {
    'data': '#3498db',     # modra
    'process': '#e74c3c',  # cervena 
    'model': '#2ecc71',    # zelena
    'output': '#f39c12',   # oranzova
    'measure': '#9b59b6',  # fialova
    'pozadi': '#ecf0f1',   # svetle seda
    'text': '#2c3e50'      # tmave modra
}

# data bloky
def pridej_data_box(x, y, sirka, vyska, nazev, popisek=""):
    box = FancyBboxPatch(
        (x, y), sirka, vyska, 
        boxstyle="round,pad=0.3", 
        fc=barvy['data'], ec="black", alpha=0.8
    )
    ax.add_patch(box)
    ax.text(x + sirka/2, y + vyska/2, nazev, 
            fontsize=12, fontweight='bold', ha='center', va='center',
            color='white')
    
    if popisek:
        ax.text(x + sirka/2, y - 0.2, popisek, 
                fontsize=9, ha='center', va='top', color=barvy['text'],
                fontweight='normal', fontstyle='italic')

# proces bloky
def pridej_proces_box(x, y, sirka, vyska, nazev, popisek=""):
    box = FancyBboxPatch(
        (x, y), sirka, vyska, 
        boxstyle="sawtooth,pad=0.3", 
        fc=barvy['process'], ec="black", alpha=0.8
    )
    ax.add_patch(box)
    ax.text(x + sirka/2, y + vyska/2, nazev, 
            fontsize=12, fontweight='bold', ha='center', va='center',
            color='white')
    
    if popisek:
        ax.text(x + sirka/2, y - 0.2, popisek, 
                fontsize=9, ha='center', va='top', color=barvy['text'],
                fontweight='normal', fontstyle='italic')

# model bloky
def pridej_model_box(x, y, sirka, vyska, nazev, popisek=""):
    box = FancyBboxPatch(
        (x, y), sirka, vyska, 
        boxstyle="round4,pad=0.3", 
        fc=barvy['model'], ec="black", alpha=0.8
    )
    ax.add_patch(box)
    ax.text(x + sirka/2, y + vyska/2, nazev, 
            fontsize=12, fontweight='bold', ha='center', va='center',
            color='white')
    
    if popisek:
        ax.text(x + sirka/2, y - 0.2, popisek, 
                fontsize=9, ha='center', va='top', color=barvy['text'],
                fontweight='normal', fontstyle='italic')

# vystup bloky
def pridej_vystup_box(x, y, sirka, vyska, nazev, popisek=""):
    box = FancyBboxPatch(
        (x, y), sirka, vyska, 
        boxstyle="rarrow,pad=0.3", 
        fc=barvy['output'], ec="black", alpha=0.8
    )
    ax.add_patch(box)
    ax.text(x + sirka/2 - 0.3, y + vyska/2, nazev, 
            fontsize=12, fontweight='bold', ha='center', va='center',
            color='white')
    
    if popisek:
        ax.text(x + sirka/2, y - 0.2, popisek, 
                fontsize=9, ha='center', va='top', color=barvy['text'],
                fontweight='normal', fontstyle='italic')

# metrika bloky
def pridej_metriku(x, y, sirka, vyska, nazev, popisek=""):
    box = FancyBboxPatch(
        (x, y), sirka, vyska, 
        boxstyle="round,pad=0.3", 
        fc=barvy['measure'], ec="black", alpha=0.8
    )
    ax.add_patch(box)
    ax.text(x + sirka/2, y + vyska/2, nazev, 
            fontsize=12, fontweight='bold', ha='center', va='center',
            color='white')
    
    if popisek:
        ax.text(x + sirka/2, y - 0.2, popisek, 
                fontsize=9, ha='center', va='top', color=barvy['text'],
                fontweight='normal', fontstyle='italic')

# sipky
def pridej_sipku(x_start, y_start, x_end, y_end, popisek=""):
    arrow = FancyArrow(
        x_start, y_start, x_end - x_start, y_end - y_start,
        width=0.05, head_width=0.2, head_length=0.2, fc='black', ec='black'
    )
    ax.add_patch(arrow)
    
    if popisek:
        # pozice popisu
        x_mid = x_start + (x_end - x_start) * 0.5
        y_mid = y_start + (y_end - y_start) * 0.5
        # vypocet uhlu sipky pro natoceni textu
        angle = np.arctan2(y_end - y_start, x_end - x_start) * 180 / np.pi
        
        # popisek sipky je natoceny ve smeru sipky
        ax.text(x_mid, y_mid, popisek, 
                fontsize=8, ha='center', va='center', color=barvy['text'],
                rotation=angle if abs(angle) < 90 else angle + 180,
                rotation_mode='anchor')

# ========== DATA FLOW DIAGRAM ==========

# vstupni data
pridej_data_box(1, 11, 3, 0.8, "CSV Vstupní data", "data-recovery.csv")
pridej_proces_box(5, 11, 2, 0.8, "nacti_data()", "pandas.read_csv")
pridej_data_box(8, 11, 1.5, 0.8, "DataFrame", "surova data")

# extrakce bodu
pridej_proces_box(5, 9.5, 2, 0.8, "najdi_body()", "regex parsing")
pridej_data_box(8, 9.5, 1.5, 0.8, "cisla_bodu", "ID klicovych bodu")

# rozdeleni dat
pridej_proces_box(5, 8, 2, 0.8, "rozdel_data()", "train_test_split")
pridej_data_box(2, 8, 2, 0.8, "train_idx", "indexy trenink")
pridej_data_box(8, 8, 2, 0.8, "test_idx", "indexy test")

# preprocessing pro kazdy bod
pridej_data_box(1, 6.5, 1.5, 0.8, "DataFrame", "surova data")
pridej_data_box(3, 6.5, 1.5, 0.8, "cisla_bodu", "ID bodu")
pridej_proces_box(5, 6.5, 2, 0.8, "zpracuj_bod()", "pro kazdy bod")
pridej_data_box(8, 6.5, 1.5, 0.8, "priznaky", "X_train, X_test")

# extrakce priznaku
pridej_proces_box(5, 5, 2, 0.8, "udel_priznaky()", "feature extraction")
pridej_data_box(8, 5, 1.5, 0.8, "X_train", "priznaky trening")
pridej_data_box(2, 5, 1.5, 0.8, "y_train", "cilove hodnoty")

# trenink
pridej_data_box(1, 3.5, 1.5, 0.8, "X_train", "priznaky")
pridej_data_box(3, 3.5, 1.5, 0.8, "y_train", "cile y/x")
pridej_proces_box(5, 3.5, 2, 0.8, "trenuj_model()", "RandomForest")
pridej_model_box(8, 3.5, 1.5, 0.8, "RF Model", "y/x modely")

# predikce
pridej_data_box(1, 2, 1.5, 0.8, "X_test", "testovaci data")
pridej_proces_box(3, 2, 2, 0.8, "model.predict()", "RF predikce")
pridej_data_box(6, 2, 1.5, 0.8, "predikce", "y_rf/x_rf")
pridej_data_box(9, 2, 0.9, 0.8, "pravda", "skutecne")

# vyhodnoceni
pridej_data_box(1, 0.5, 1, 0.8, "predikce", "y/x")
pridej_data_box(2.5, 0.5, 1, 0.8, "pravda", "y/x")
pridej_proces_box(4, 0.5, 2, 0.8, "spocti_chyby()", "MSE/stats")
pridej_metriku(6.5, 0.5, 1.5, 0.8, "metrika", "MSE/zlepšení")
pridej_vystup_box(8.5, 0.5, 1.5, 0.8, "CSV", "výsledky")

# sipky
pridej_sipku(4, 11.4, 5, 11.4, "CSV vstup")
pridej_sipku(7, 11.4, 8, 11.4, "DataFrame")
pridej_sipku(8.75, 11, 8.75, 9.9, "poskytuje sloupce")
pridej_sipku(6, 11, 6, 9.9, "df -> najdi_body")
pridej_sipku(8.75, 9.5, 8.75, 8.4, "pouzito v")
pridej_sipku(6, 9.5, 6, 8.4, "df -> rozdel_data")
pridej_sipku(5, 8, 4, 8, "trenovaci data")
pridej_sipku(7, 8, 8, 8, "testovaci data")

pridej_sipku(2.5, 6.5, 5, 6.5, "zpracovani dat")
pridej_sipku(6, 6.5, 7, 6.5, "extrahovano")
pridej_sipku(8.75, 6.5, 8.75, 5.4, "pro trenink")
pridej_sipku(6, 6.5, 6, 5.4, "df -> udel_priznaky")
pridej_sipku(5, 5, 3.5, 5, "cilove hodnoty")
pridej_sipku(7, 5, 8, 5, "trenovaci X")
pridej_sipku(1.75, 5, 1.75, 3.9, "X pro trenink")
pridej_sipku(3.75, 5, 3.75, 3.9, "Y pro trenink")
pridej_sipku(3.5, 3.5, 5, 3.5, "data -> model")
pridej_sipku(7, 3.5, 8, 3.5, "vytvoreny model")
pridej_sipku(8.75, 3.5, 8.75, 2.4, "pouziti modelu")
pridej_sipku(1.75, 3.5, 1.75, 2.4, "testovaci X")
pridej_sipku(5, 2, 6, 2, "generuje predikce")
pridej_sipku(7.5, 2, 9, 2, "pro porovnani")
pridej_sipku(1.5, 2, 1.5, 0.9, "predikce")
pridej_sipku(9.5, 2, 9.5, 1.3, "skutecne")
pridej_sipku(9.5, 1.3, 3, 0.9, "pro vyhodnoceni")
pridej_sipku(6, 0.5, 6.5, 0.5, "vypocet")
pridej_sipku(8, 0.5, 8.5, 0.5, "ulozeni")

# nazev grafu
plt.title("Datový tok při predikci bodů", fontsize=16, pad=20)

# popis
plt.figtext(0.5, 0.02, "Program pro predikci klicovych bodu - datovy tok", 
            ha="center", fontsize=10, fontstyle='italic')

plt.savefig("diagrams/data_flow.png", dpi=300, bbox_inches='tight')
print("Graf datového toku uložen do diagrams/data_flow.png") 