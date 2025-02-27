from __future__ import annotations as _annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Optional
from prettytable import PrettyTable

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

load_dotenv("../.env")


# Define dependencies for travel preferences
@dataclass
class TravelPreferences:
    location_type: Optional[str] = None  # Museums, Parks, etc.
    num_places: int = 5  # Default number of locations to return
    current_date: date = date.today()  # Automatically set to today's date


# Define a structured response model for a tourist place
class TouristPlace(BaseModel):
    name: str  # Enforces name as a string
    description: str  # Ensures AI provides meaningful descriptions
    zip_code: int  # Enforces zipcode as a integer
    entry_fee: Optional[float] = Field(None, description="Entry fee in USD, if applicable")
    rating: float = Field(..., ge=0.0, le=5.0, description="Average rating from 0.0 to 5.0")


# Initialize AI model with structured validation
travel_agent = Agent(
    "groq:llama-3.3-70b-versatile",
    model_settings={"temperature": 0.1},
    deps_type=TravelPreferences,     # Passing dependency to the AI agent
    result_type=List[TouristPlace],  # Enforces structured output
)


@travel_agent.system_prompt
def generate_system_prompt(ctx: RunContext[TravelPreferences]) -> str:
    """
    Dynamically generates a system prompt based on user travel preferences.
    """
    month: str = ctx.deps.current_date.strftime("%B")  # Get current month
    return (
        f"You are an AI-powered travel guide specializing in {ctx.deps.location_type or 'all types of locations'}.\n"
        f"The current month is {month}.\n"
        f"Recommend the best tourist places to visit this month, considering seasonality.\n"
        f"Provide structured tourist recommendations, limited to {ctx.deps.num_places} places."
    )


if __name__ == "__main__":
    while True:
        city_name: str = str(input("Enter name of the city (or enter q to quite): ")).lower()
        # User Input: New York City

        if city_name == "q":
            break
        else:
            location_type: str = str(input("Type of places interested in (for example: Museums, Parks etc.): ")).lower()
            # User Input: Museums

            num_places: int = int(input("Number of places to visit: "))
            # User Input: 3

            # Inject preferences: Fetch number of places based on user preference best suited for the current month
            deps_TravelPreferences = TravelPreferences(
                location_type=location_type,
                num_places=num_places,
                current_date=date.today(),
            )

            result = travel_agent.run_sync(city_name, deps=deps_TravelPreferences)

            # Print system prompt
            print(result._all_messages[0].parts[0])
            # Output: SystemPromptPart(content='You are an AI-powered travel guide specializing in museums.
            # The current month is February.
            # Recommend the best tourist places to visit this month, considering seasonality.
            # Provide structured tourist recommendations, limited to 3 places.',
            # dynamic_ref=None, 
            # part_kind='system-prompt')

            print("\nAgent Output: \n")

            # Converting result into table format
            result_table = PrettyTable()
            result_table.field_names = ["Name", "Description", "Location - zip_code", "Entry Fee (USD)", "Rating",]

            for place in result.data:
                result_table.add_row(
                    [
                        place.name,
                        place.description,
                        place.zip_code,
                        "Free"
                        if place.entry_fee is None
                        else f"${place.entry_fee:.2f}",
                        place.rating,
                    ]
                )

            print(result_table)

# Output:
# +------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+--------+
# |                Name                |                                                                Description                                                                | Location - zip_code | Entry Fee (USD) | Rating |
# +------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+--------+
# |   The Metropolitan Museum of Art   |          One of the world's largest and most renowned museums, with a vast collection of art and artifacts from around the world.         |        10028        |      $25.00     |  4.8   |
# | American Museum of Natural History |    A museum showcasing a vast collection of natural history specimens and artifacts, including dinosaur fossils and a giant blue whale.   |        10024        |      $22.00     |  4.5   |
# |    Museum of Modern Art (MoMA)     | One of the most influential modern art museums in the world, with a collection of works by artists such as Van Gogh, Picasso, and Warhol. |        10019        |      $25.00     |  4.7   |
# +------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------------+---------------------+-----------------+--------+

