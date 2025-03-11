import json
import os
from datetime import datetime

class WeatherDataLogger:
    """Ukládá získaná data o počasí do JSON souboru."""
    FILE_NAME = "weather_data.json"

    def __init__(self):
        """Načte existující soubor nebo vytvoří prázdný JSON, pokud neexistuje."""
        if not os.path.exists(self.FILE_NAME):
            with open(self.FILE_NAME, "w") as file:
                json.dump({}, file)

    def update_data(self, city, temperature):
        """Přidá nebo aktualizuje záznam pro dané město s časem."""
        now = datetime.now().strftime("%Y-%m-%d %H:00") # Zaokrouhlení na celou hodinu
        # Načtení existujících dat
        with open(self.FILE_NAME, "r") as file:
            data = json.load(file)
        # Přidání/aktualizace města a teploty
        if city not in data:
            data[city] = {} # Vytvoříme záznam pro město
            data[city][now] = temperature
        # Uložení zpět do souboru
        with open(self.FILE_NAME, "w") as file:
            json.dump(data, file, indent=2)
            print(f"Data for {city} updated: {temperature}°C at {now}")