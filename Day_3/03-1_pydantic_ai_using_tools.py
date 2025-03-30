from __future__ import annotations as _annotations


import requests
import os
from dataclasses import dataclass
from datetime import date
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table

from pydantic import BaseModel, Field, SecretStr
from pydantic_ai import Agent, RunContext
from duckduckgo_search import DDGS

import logfire
from dotenv import load_dotenv


load_dotenv("../.env")

logfire.configure()

# Define dependencies
@dataclass
class Deps:
    current_date: date = date.today()  # Automatically set to today's date
    alpha_vantage_api_key: SecretStr | None = None  # API key for WeatherStack

# Define a structured response model for stock details
class StockDetails(BaseModel):
    company_name: str = Field(description="Name of a company")
    ticker: str = Field(description="Stock ticker symbol for a company")
    company_description: str = Field(description="Company Description")
    sector: str = Field(description="Sector in which the company operates")
    industry: str = Field(description="Industry in which the company operates")
    market_capitalization: str = Field(description="Market Capitalization of the company")
    stock_exchange: str = Field(description="Stock exchange on which the company is listed")
    current_stock_price: float = Field(description="Current stock price of the company")
    v_52_week_high: float = Field(description="52 Week high stock price of the company")
    v_52_week_low: float = Field(description="52 week low stock price of the company")

stock_market_agent = Agent(
    "google-gla:gemini-2.5-pro-exp-03-25", 
    model_settings={"temperature": 0},
    result_type=StockDetails,
    deps_type=Deps,
    system_prompt="""As an AI agent with stock market knowledge, you will provide up-to-date stock market related details about the company.
        Follow the below steps and guidelines: 
        step 1: Use the `get_stock_ticker_symbol` tool to get the stock ticker symbol for the company, 
        step 2: Use the `get_current_stock_price` tool to fetch the current stock price of a given company, 
        step 3: Use the `get_company_overview_and_financials` tool to get company details and other financial data like sector, industry, market capitalization, stock exchange, 52 week high and low price. 
        step 4: Use the information returned by these tools to provide the stock details about the company. 
        You must always provide the most accurate and up-to-date data using the above tools only. Do not fall back to generic knowledge or assumptions.""",
    retries=3,
)


# Tool without run context
@stock_market_agent.tool_plain(retries=3)
async def get_stock_ticker_symbol(company_name: str) -> List[Dict[str, str]]:
    """Get Stock ticker symbol for a company

    Args:
        company_name: Name of the company
    """

    logfire.info(f"DuckDuckGo Search: {company_name}")

    results: List[Dict[str, str]] = DDGS().text(f"What is stock ticker symbol for {company_name}", max_results=5)
    return results


# Tool with run context
@stock_market_agent.tool
async def get_current_stock_price(ctx: RunContext[Deps], ticker: str) -> Dict[str, Any]:
    """Fetch the current stock price of a given company

    Args:
        ticker: Stock ticker symbol for a company
    """

    logfire.info(f"Get current stock price of the company from alphavantage: {ticker}")
    
    url: str = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ctx.deps.alpha_vantage_api_key}"
    stock_price = requests.get(url)
    # print(stock_price.text)
    return stock_price.text

# Tool with run context
@stock_market_agent.tool
async def get_company_overview_and_financials(ctx: RunContext[Deps], ticker: str) -> Dict[str, Any]:
    """Get company details and other financial data

    Args:
        ticker: Stock ticker symbol for a company
    """

    logfire.info(f"Get company details and other financial data of the company from alphavantage: {ticker}")
    
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ctx.deps.alpha_vantage_api_key}"
    company_overview = requests.get(url)
    # print(company_overview.text)
    return company_overview.text


if __name__ == "__main__":
    alpha_vantage_api_key: SecretStr = os.getenv("ALPHA_VANTAGE_API_KEY")
    # Run the agent
    result = stock_market_agent.run_sync('Provide details about company Apple', deps=Deps(alpha_vantage_api_key=alpha_vantage_api_key))
    # print(result.data)

    # Create a console object
    console = Console()

    # Create a table
    table = Table(title="Company Information", show_header=False)
    table.add_column("Attribute", style="bold")
    table.add_column("Value")

    # Populate table with dictionary data
    for key, value in result.data:
        table.add_row(key, str(value))

    # Print the table
    console.print(table)