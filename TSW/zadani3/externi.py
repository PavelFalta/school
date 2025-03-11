import os
from dotenv import load_dotenv
from logger import WeatherDataLogger
from weather_stub import WeatherService


# Načtení proměnných z .env souboru
load_dotenv()
API_KEY = os.getenv("API_KEY")



# Testovací běh
if __name__ == "__main__":
    city = "Prague"
    service = WeatherService(API_KEY=API_KEY)
    response = service.get_weather(city)
    temperature = response["main"]["temp"]
    print(temperature)

    logger = WeatherDataLogger()
    logger.update_data(city, temperature)