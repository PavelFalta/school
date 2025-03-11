import requests


class WeatherService:
    """Služba pro získání počasí pomocí API klíče uloženého v .env."""
    def get_weather(self, city, API_KEY):
        """Vrátí počasí pro dané město na základě API volání."""
        if not API_KEY:
            raise ValueError("API klíč není nastaven.")
        
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            ret