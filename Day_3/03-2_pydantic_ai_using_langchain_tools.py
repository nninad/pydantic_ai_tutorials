from __future__ import annotations as _annotations

from typing import Dict, List

import logfire
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from pydantic_ai import Agent

load_dotenv("../.env")

logfire.configure()

async def get_stock_ticker_symbol(company_name: str) -> List[Dict[str, str]]:
    """Stock ticker symbol for a company

    Args:
        company_name: Name of the company
    """

    logfire.info(f"DuckDuckGo Search: {company_name}")

    results: List[Dict[str, str]] = DDGS().text(f"What is stock ticker symbol for {company_name}", max_results=5)
    return results


async def get_latest_financial_news(ticker: str) -> str:
    """Fetch the current financial news of a given company

    Args:
        ticker: Stock ticker symbol for a company
    """

    logfire.info(f"Yahoo Finance News: {ticker}")

    finance_news = YahooFinanceNewsTool().run(ticker)
    
    print(finance_news)
    return finance_news


finance_agent = Agent(
    # "groq:llama-3.3-70b-versatile",
    "google-gla:gemini-2.5-pro-exp-03-25",  # Using Gemini Pro model for better performance
    model_settings={"temperature": 0.1},
    result_type=str,
    system_prompt="""As an AI agent with stock market knowledge, you will provide up-to-date stock market related details about the company.
        Follow the below steps and guidelines:
        step 1: Use the `get_stock_ticker_symbol` tool to get the stock ticker symbol for the company.
        step 2: Use the `get_latest_financial_news` tool to fetch the current financial news about the company.
        You must always provide the most accurate and up-to-date data using the above tools only. Do not fall back to generic knowledge or assumptions.'""",
        retries=3,
        tools=[get_stock_ticker_symbol, get_latest_financial_news],
)

if __name__ == "__main__":
    # Run the agent
    result = finance_agent.run_sync('Provide latest news about Google')
    print(result.data)