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
def web_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news"] = "general",
) -> dict:
    """Search the web for current information.

    Args:
        query: The search query (be specific and detailed)
        max_results: Number of results to return (default: 5)
        topic: "general" for most queries, "news" for current events

    Returns:
        Search results with titles, URLs, and content excerpts.
    """
    try:
        client = TavilySearch(tavily_api_key=settings.TAVILY_API_KEY)
        return client.search(query, max_results=max_results, topic=topic)
    except Exception as e:
        return {"error": f"Search failed: {e}"}


@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"
