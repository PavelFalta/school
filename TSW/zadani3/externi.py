import os
from dotenv import load_dotenv
from logger import WeatherDataLogger
from weather_mock import WeatherService


if __name__ == "__main__":
    load_dotenv()

    API_KEY = os.getenv("API_KEY")
    city = "Prague"

    service = WeatherService(API_KEY=API_KEY)
    response = service.get_weather(city)

    temperature = response["main"]["temp"]

    logger = WeatherDataLogger()
    logger.update_data(city, temperature)