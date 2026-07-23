# weather_server.py
# Transport protocol: sse (server-sent-events = remote client connects to server via HTTP)
#
# Uses the free Open-Meteo API (no API key required) to fetch real current
# weather conditions: temperature, humidity, and precipitation (rainfall).

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather", host="127.0.0.1", port=8000)

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes -> human-readable condition
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


async def _geocode(location: str):
    """Resolve a city name to (lat, lon, resolved_name, country)."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(GEOCODING_URL, params={"name": location, "count": 1})
        resp.raise_for_status()
        results = resp.json().get("results")
        if not results:
            return None
        top = results[0]
        return top["latitude"], top["longitude"], top.get("name", location), top.get("country", "")


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get the current weather for a location, including temperature, rainfall
    (precipitation), humidity, and overall condition.

    Args:
        location: City name, e.g. "New York" or "London".
    """
    geo = await _geocode(location)
    if geo is None:
        return f"Could not find a location matching '{location}'."

    lat, lon, resolved_name, country = geo

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            FORECAST_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
                "timezone": "auto",
            },
        )
        resp.raise_for_status()
        current = resp.json().get("current", {})

    temperature = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    precipitation = current.get("precipitation")
    weather_code = current.get("weather_code")
    condition = WEATHER_CODES.get(weather_code, "Unknown")

    return (
        f"Weather in {resolved_name}, {country}: {condition}. "
        f"Temperature: {temperature}°C. "
        f"Humidity: {humidity}%. "
        f"Rainfall (precipitation): {precipitation} mm."
    )


if __name__ == "__main__":
    # Run this file directly and keep it running BEFORE starting main.py,
    # since SSE is an HTTP server the client connects to remotely
    # (unlike math_server, which the client launches itself over stdio).
    mcp.run(transport="sse")