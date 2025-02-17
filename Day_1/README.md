## Type-safe and Structured Responses: 
In modern AI-driven applications, ensuring **structured and type-safe outputs** is critical. When AI models generate responses, they often return free-form text/unstructured results, which can lead to inconsistencies, incorrect data types, or missing information. PydanticAI ensures AI-generated responses adhere to specific data types, preventing inconsistencies like numbers in place of strings or missing fields. It uses Pydantic models to define schemas, making sure the AI returns well-formed JSON-like outputs instead of free-text responses.

### Objective

We will build an AI agent that provides details about three famous tourist destinations, ensuring the response is structured and type-safe.

### Step 1: Install Dependencies

```python
uv init
uv add pydantic-ai python-dotenv prettytable
```

### Step 2: Define a Structured Output Model

Using Pydantic, we define a strict schema for AI-generated responses:

```python
# Define a structured response model for a tourist place
class TouristPlace(BaseModel):
    name: str                # Enforces name as a string
    description: str         # Ensures AI provides meaningful descriptions
    zip_code: int            # Enforces zipcode as a integer
    best_time_to_visit: str  # AI must specify the best time to visit
    entry_fee: Optional[float] = Field(None, description="Entry fee in USD, if applicable")
    rating: float = Field(..., ge=0.0, le=5.0, description="Average rating from 0.0 to 5.0")
```

### Step 3: Initialize Agent

We will initialize an AI agent with structured validation using **Groq**. If you prefer a **local model using Ollama**, use the alternative code block provided.

**Using Groq**

```python
# Initialize AI model with structured validation
travel_agent = Agent(
    "groq:llama-3.3-70b-versatile",
    model_settings={"temperature": 0.1},
    result_type=List[TouristPlace],      # Enforces structured output
    system_prompt="Provide 3 famous tourist places in the given city with descriptions, best time to visit, entry fee (if any), and average visitor rating.",
)
```

**Using Ollama (Local Model)**

```python
# Initialize AI model with structured validation
ollama_model = OpenAIModel(model_name='llama3.2', base_url='http://localhost:11434/v1')

travel_agent = Agent(
    model=ollama_model,
    model_settings={"temperature": 0.1},
    result_type=List[TouristPlace],      # Enforces structured output
    system_prompt="Provide 3 famous tourist places in the given city with descriptions, best time to visit, entry fee (if any), and average visitor rating.",
)
```

### Step 4: Run the AI Agent

Now, let’s ask the AI agent about 3 famous tourist places in New York City.

```python
result = travel_agent.run_sync("New York City")
print(f"Agent Output: \n {result}")
```

### Step 5: Result

Instead of free-form text, the AI agent returned **structured data**:

```json
[
    {
        "name": "The Statue of Liberty",
        "description": "A iconic symbol of freedom and democracy",
        "best_time_to_visit": "Summer",
        "entry_fee": 21.5,
        "rating": 4.8,
        "zip_code": 10004
    },
    {
        "name": "Central Park",
        "description": "A large public park in Manhattan",
        "best_time_to_visit": "Autumn",
        "entry_fee": null,
        "rating": 4.7,
        "zip_code": 10021
    },
    {
        "name": "The Metropolitan Museum of Art",
        "description": "One of the world's largest and most renowned museums",
        "best_time_to_visit": "Spring",
        "entry_fee": 25,
        "rating": 4.9,
        "zip_code": 10028
    }
]
```

### How PydanticAI Ensures Type-Safety and Structure

### ✅ Prevents incorrect data types

- `entry_fee` must be a float (or `None`), preventing incorrect values like `"Free"`.
- `rating` must be between 0.0 and 5.0, ensuring no out-of-range values.

### ✅ Enforces a Strict Schema

- AI must provide all required fields (`name`, `description`, `location`, etc.).

### ✅ Automatic error handling

- If AI generates an invalid response, PydanticAI raises validation errors before processing.

### Example: AI Agent Returns Invalid Data

```python
bad_response= {
            "name": "Statue of Liberty",
            "description": null,
            "best_time_to_visit": "Spring and Fall",
            "entry_fee": "Free",     # ❌ Should be float or None
            "rating": 6.2,           # ❌ Should be ≤ 5.0
            "zip_code": 10028.90     # ❌ Should be integer
        }

validated_response = TouristPlace(**bad_response)  # Raises error
```

### Error Output:

```less
description
  Input should be a valid string [type=string_type, input_value=<function null at 0x7f9bca26f400>, input_type=function]
    For further information visit https://errors.pydantic.dev/2.10/v/string_type
zip_code
  Input should be a valid integer, got a number with a fractional part [type=int_from_float, input_value=10028.9, input_type=float]
    For further information visit https://errors.pydantic.dev/2.10/v/int_from_float
entry_fee
  Input should be a valid number, unable to parse string as a number [type=float_parsing, input_value='Free', input_type=str]
    For further information visit https://errors.pydantic.dev/2.10/v/float_parsing
rating
  Input should be less than or equal to 5 [type=less_than_equal, input_value=6.2, input_type=float]
    For further information visit https://errors.pydantic.dev/2.10/v/less_than_equal
```

### Final Thoughts: Why Use PydanticAI?

PydanticAI is a **game-changer** for AI applications that require structured and validated outputs.

✅ **Improves reliability** of AI responses

✅ **Prevents malformed data** in AI-driven apps

✅ **Enhances automation** in data pipelines

✅ **Reduces debugging time** with automatic validation

If you’re working on AI-powered applications that require structured responses, PydanticAI ensures **accuracy, consistency, and type safety**.
