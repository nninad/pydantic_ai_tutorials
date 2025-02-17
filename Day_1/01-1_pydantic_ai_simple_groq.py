from __future__ import annotations as _annotations

from typing import List, Optional
from prettytable import PrettyTable

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent

load_dotenv("../.env")


# Define a structured response model for a tourist place
class TouristPlace(BaseModel):
    name: str  # Enforces name as a string
    description: str  # Ensures AI provides meaningful descriptions
    zip_code: int  # Enforces zipcode as a integer
    best_time_to_visit: str  # AI must specify the best time to visit
    entry_fee: Optional[float] = Field(
        None, description="Entry fee in USD, if applicable"
    )
    rating: float = Field(
        ..., ge=0.0, le=5.0, description="Average rating from 0.0 to 5.0"
    )


# Initialize AI model with structured validation
travel_agent = Agent(
    "groq:llama-3.3-70b-versatile",
    model_settings={"temperature": 0.1},
    result_type=List[TouristPlace],  # Enforces structured output
    system_prompt="Provide 3 famous tourist places in the given city with descriptions, best time to visit, entry fee (if any), and average visitor rating.",
)

result = travel_agent.run_sync("New York City")
print("Agent Output: \n")

print(f"Data: \n {result.data} \n\n")
print(f"Usage: \n {result.usage()} \n\n")


# Converting result into table format
result_table = PrettyTable()
result_table.field_names = [
    "Name",
    "Description",
    "Location - zip_code",
    "Best Time to Visit",
    "Entry Fee (USD)",
    "Rating",
]

for place in result.data:
    result_table.add_row(
        [
            place.name,
            place.description,
            place.zip_code,
            place.best_time_to_visit,
            "Free" if place.entry_fee is None else f"${place.entry_fee:.2f}",
            place.rating,
        ]
    )

print(result_table)
