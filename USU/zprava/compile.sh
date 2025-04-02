#!/bin/bash

# Přejít do adresáře zprávy
cd "$(dirname "$0")"

# Zkontrolovat, zda jsou k dispozici potřebné nástroje
if ! command -v pdflatex &> /dev/null; then
    echo "Error: pdflatex není nainstalován. Prosím nainstalujte TeX Live nebo jinou LaTeX distribuci."
    exit 1
fi

echo "Začínám kompilaci LaTeX zprávy..."

# Kompilace LaTeX dokumentu (dvakrát pro správné reference)
pdflatex -interaction=nonstopmode zprava.tex
pdflatex -interaction=nonstopmode zprava.tex

# Vyčistit dočasné soubory
echo "Čistím dočasné soubory..."
rm -f *.aux *.log *.out *.toc

echo "Kompilace dokončena. Výsledný PDF soubor: zprava.pdf"
echo "Můžete jej otevřít pomocí příkazu: xdg-open zprava.pdf" 