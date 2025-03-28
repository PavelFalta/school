from weather_stub import WeatherService
import pytest

API_KEY = 123456
city = "Prague"

def test_get_weather():
    service = WeatherService(API_KEY=API_KEY)
    response = service.get_weather(city)
    assert response["name"] == city
    assert response["main"]["temp"] == 9.99

