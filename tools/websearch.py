from typing import Literal

from langchain_tavily import TavilySearch
from core.config import settings
from langchain.tools import tool


@tool
def internet_search(
        query: str,
        max_results: int = 5,
        topic: Literal["general", "news", "finance"] = "general",
        include_raw_content: bool = False,
):
    """Run a web search"""
    tavily_client = TavilySearch(tavily_api_key=settings.TAVILY_API_KEY)
    return tavily_client.invoke(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"
