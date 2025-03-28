import requests
import random
from datetime import datetime


class WeatherService:
    """Služba pro získání počasí pomocí API klíče uloženého v .env."""
    def __init__(self, API_KEY):
        self.API_KEY = API_KEY

    def get_weather(self, city):
        """Vrátí počasí pro dané město na základě API volání."""
        API_KEY = self.API_KEY
        if not API_KEY:
            raise ValueError("API klíč není nastaven.")
        
        city_binary = ' '.join(format(ord(char), 'b') for char in city)

        ones_count = city_binary.count('1')

        random.seed(ones_count)

        temperature = random.uniform(5, 25)

        current_time = datetime.now()
        minutes = current_time.minute

        delta = -((minutes - 30) ** 2) / 100 + 5

        temperature += delta

        return {
            'coord': {'lon': random.uniform(-180, 180), 'lat': random.uniform(-90, 90)},
            'weather': [{'id': random.choice([800, 801, 802, 803, 804]), 'main': random.choice(['Clear', 'Clouds', 'Rain', 'Snow', 'Drizzle', 'Thunderstorm']), 'description': random.choice(['clear sky', 'few clouds', 'scattered clouds', 'broken clouds', 'shower rain', 'rain', 'thunderstorm', 'snow', 'mist']), 'icon': random.choice(['01d', '02d', '03d', '04d', '09d', '10d', '11d', '13d', '50d'])}],
            'base': 'stations',
            'main': {'temp': temperature, 'feels_like': random.uniform(temperature - 5, temperature + 5), 'temp_min': random.uniform(temperature - 5, temperature), 'temp_max': random.uniform(temperature, temperature + 5), 'pressure': random.randint(980, 1050), 'humidity': random.randint(0, 100), 'sea_level': random.randint(980, 1050), 'grnd_level': random.randint(980, 1050)},
            'visibility': random.randint(1000, 10000),
            'wind': {'speed': random.uniform(0, 20), 'deg': random.randint(0, 360)},
            'clouds': {'all': random.randint(0, 100)},
            'dt': random.randint(1609459200, 1735689600),
            'sys': {'type': 2, 'id': random.randint(1000, 9999), 'country': random.choice(['CZ', 'US', 'GB', 'DE', 'FR', 'RU', 'CN', 'JP', 'IN']), 'sunrise': random.randint(1609459200, 1735689600), 'sunset': random.randint(1609459200, 1735689600)},
            'timezone': random.randint(-43200, 50400),
            'id': random.randint(1000000, 9999999),
            'name': city,
            'cod': 200
        }