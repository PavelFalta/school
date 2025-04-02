import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrow, FancyBboxPatch

# nastaveni velikosti a stylu
plt.figure(figsize=(14, 10))
plt.style.use('ggplot')

ax = plt.gca()
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# barvy
barvy = {
    'hlavni': '#3498db',  # modra
    'nacteni': '#e74c3c',  # cervena 
    'zpracovani': '#2ecc71',  # zelena
    'predikce': '#f39c12',  # oranzova
    'hodnoceni': '#9b59b6',  # fialova
    'pozadi': '#ecf0f1',  # svetle seda
    'text': '#2c3e50'  # tmave modra
}

# funkce pro box
def pridej_box(x, y, sirka, vyska, barva, nazev, popisek=""):
    box = FancyBboxPatch(
        (x, y), sirka, vyska, 
        boxstyle="round,pad=0.3", 
        fc=barva, ec="black", alpha=0.8
    )
    ax.add_patch(box)
    ax.text(x + sirka/2, y + vyska/2, nazev, 
            fontsize=12, fontweight='bold', ha='center', va='center',
            color=barvy['text'])
    
    if popisek:
        ax.text(x + sirka/2, y - 0.2, popisek, 
                fontsize=10, ha='center', va='top', color=barvy['text'],
                fontweight='normal', fontstyle='italic')

# sipky
def pridej_sipku(x_start, y_start, x_end, y_end):
    arrow = FancyArrow(
        x_start, y_start, x_end - x_start, y_end - y_start,
        width=0.05, head_width=0.2, head_length=0.2, fc='black', ec='black'
    )
    ax.add_patch(arrow)


# hlavni program
pridej_box(4, 8.5, 2, 1, barvy['hlavni'], "predikuj_body()", "hlavni funkce")

# nacteni dat
pridej_box(1, 7, 2, 0.8, barvy['nacteni'], "nacti_data()", "csv -> pandas df")
pridej_box(4, 7, 2, 0.8, barvy['nacteni'], "najdi_body()", "regex hledani")
pridej_box(7, 7, 2, 0.8, barvy['nacteni'], "rozdel_data()", "train/test split")

# preprocessing
pridej_box(1, 5.5, 2, 0.8, barvy['zpracovani'], "zpracuj_bod()", "pro kazdy bod")
pridej_box(4, 5.5, 2, 0.8, barvy['zpracovani'], "udel_priznaky()", "vytvori vektory")
pridej_box(7, 5.5, 2, 0.8, barvy['zpracovani'], "udel_zakladni\npredikce()", "baseline")

# trenovani
pridej_box(1, 4, 2, 0.8, barvy['predikce'], "trenuj_model()", "RandomForest")
pridej_box(4, 4, 2, 0.8, barvy['predikce'], "y_model", "pro y souradnice")
pridej_box(7, 4, 2, 0.8, barvy['predikce'], "x_model", "pro x souradnice")

# vyhodnoceni
pridej_box(1, 2.5, 2, 0.8, barvy['hodnoceni'], "spocti_chyby()", "per-point errors")
pridej_box(4, 2.5, 2, 0.8, barvy['hodnoceni'], "spocti_mse()", "celkova metrika")
pridej_box(7, 2.5, 2, 0.8, barvy['hodnoceni'], "uloz_predikce()", "ulozi .csv")

# sipky
pridej_sipku(5, 8.5, 5, 7.8)  # hlavni -> nacteni
pridej_sipku(3, 7.4, 4, 7.4)  # nacteni -> najdi
pridej_sipku(6, 7.4, 7, 7.4)  # najdi -> rozdel

pridej_sipku(2, 7, 2, 6.3)    # nacteni -> zpracuj
pridej_sipku(5, 6.7, 5, 6.3)  # zpracovani -> udel
pridej_sipku(8, 6.7, 8, 6.3)  # zpracovani -> udel_zakladni

pridej_sipku(2, 5.5, 2, 4.8)  # zpracuj -> trenuj
pridej_sipku(3, 4.4, 4, 4.4)  # trenuj -> y_model
pridej_sipku(6, 4.4, 7, 4.4)  # y_model -> x_model

pridej_sipku(2, 4, 2, 3.3)    # trenuj -> spocti_chyby
pridej_sipku(3, 2.9, 4, 2.9)  # spocti_chyby -> spocti_mse
pridej_sipku(6, 2.9, 7, 2.9)  # spocti_mse -> uloz

# nazev grafu
plt.title("Struktura programu pro predikci bodů", fontsize=16, pad=20)

# popis
plt.figtext(0.5, 0.02, "Program pro predikci klicovych bodu - architektura", 
            ha="center", fontsize=10, fontstyle='italic')

plt.savefig("diagrams/program_structure.png", dpi=300, bbox_inches='tight')
print("Graf struktury programu uložen do diagrams/program_structure.png") 