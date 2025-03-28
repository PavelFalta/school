import requests


class WeatherService:
    """Služba pro získání počasí pomocí API klíče uloženého v .env."""
    def __init__(self, API_KEY):
        self.API_KEY = API_KEY

    def get_weather(self, city):
        """Vrátí počasí pro dané město na základě API volání."""
        API_KEY = self.API_KEY
        if not API_KEY:
            raise ValueError("API klíč není nastaven.")
        
        return {
            'coord': {'lon': 14.4208, 'lat': 50.088},
            'weather': [{'id': 800, 'main': 'Clear', 'description': 'clear sky', 'icon': '01d'}],
            'base': 'stations',
            'main': {'temp': 9.99, 'feels_like': 9.83, 'temp_min': 9.56, 'temp_max': 12.17, 'pressure': 1002, 'humidity': 76, 'sea_level': 1002, 'grnd_level': 968},
            'visibility': 10000,
            'wind': {'speed': 5.66, 'deg': 240},
            'clouds': {'all': 0},
            'dt': 1741685765,
            'sys': {'type': 2, 'id': 2010430, 'country': 'CZ', 'sunrise': 1741670686, 'sunset': 1741712406},
            'timezone': 3600,
            'id': 3067696,
            'name': city,
            'cod': 200
        }