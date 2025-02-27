from __future__ import annotations as _annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Any
from datetime import datetime
from tabulate import tabulate


import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from pydantic_ai import Agent, RunContext, UserError

load_dotenv("../.env")


# Define dependencies
@dataclass
class WeatherDeps:
    current_date: date = date.today()  # Automatically set to today's date
    weatherstack_api_key: SecretStr | None = None  # API key for WeatherStack


# Define a structured response model for weather details
class WeatherDetails(BaseModel):
    location_name: str = Field(description="Name of the location")
    local_time: datetime = Field(description="Local time at the location.")
    coordinates: List[float] = Field(description="Latitude and Longitude of the location")
    weather_descriptions: str = Field(description="Description of the current weather like clear, Sunny, Rainy, etc.")
    temparature: float = Field(description="Temparature")
    feels_like: float = Field(description="Feels like temparature")
    Precipitation: float = Field(description="Amount of precipitation in mm.")


# Initialize AI model with structured validation
weather_agent = Agent(
    "groq:llama-3.3-70b-versatile",
    model_settings={"temperature": 0.1},
    result_type=WeatherDetails,  # Enforces structured output
    deps_type=WeatherDeps,
    system_prompt="As a weather agent, provide detailed current weather information for any location requested, including temperature, conditions (clear, sunny, cloudy, rainy), and any relevant weather alerts, when asked. Always provide the most accurate and up-to-date data using `get_current_weather_details` tool.",
)


@weather_agent.tool
def get_current_weather_details(ctx: RunContext[WeatherDeps], city_name: str) -> Dict[str, Any]:
    """
    Retrieve current weather details for the specified city.

    Args:
        city_name: Name of the city
    """

    if ctx.deps.weatherstack_api_key is None:
        raise UserError("WeatherStack API key is required to get weather details")

    # Build API request
    url = f"https://api.weatherstack.com/current?access_key={ctx.deps.weatherstack_api_key}"
    querystring: dict[str, str] = {"query": city_name}

    # Fetch weather data
    response = requests.get(url, params=querystring)

    # Convert the JSON string to a Python dictionary
    # data: Dict[str, Any] = response.json()

    return response.json()


if __name__ == "__main__":
    while True:
        city_name: str = str(input("Enter name of the city (or enter q to quite): ")).lower()
        # User Input: New York City

        if city_name == "q":
            break
        else:
            # Fetch API key from environment variables
            weatherstack_api_key: SecretStr = os.getenv("WEATHERSTACK_API_KEY")

            # Prepare dependencies
            deps = WeatherDeps(
                current_date=date.today(),
                weatherstack_api_key=weatherstack_api_key,
            )

            # Run the agent
            # city_name = "New York City"
            result = weather_agent.run_sync(city_name, deps=deps)

            # Display results
            print("\nAgent Output: \n")
            print(tabulate(result.data))

# Agent Output:
# --------------------  -------------------
# location_name         New York City
# local_time            2025-02-23 20:11:00
# coordinates           [40.714, -74.006]
# weather_descriptions  Clear
# temparature           7.0
# feels_like            6.0
# Precipitation         0.0
# --------------------  -------------------
