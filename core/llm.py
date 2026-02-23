from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openrouter import ChatOpenRouter
from core.config import settings


def get_model_openai(stop_args: List[str] = None):
    return ChatOpenAI(
        model_name="gpt-5.2-pro",
        api_key=settings.LLM_API_KEY
    )


def get_model_open_router():
    return ChatOpenRouter(
        model_name="anthropic/claude-sonnet-4.6",
        api_key=settings.OPENROUTER_API_KEY
    )


def get_model_anthropic(stop_args: List[str] = None):
    return ChatAnthropic(
        model_name="claude-sonnet-4-5",
        api_key=settings.ANTHROPIC_API_KEY,
        timeout=60.0,
        stop=stop_args,
        streaming=True,
        max_tokens_to_sample=32768
    )


def get_model_google_genai():
    return ChatGoogleGenerativeAI(
        model_name="gemini-3.1-pro-preview",
        api_key=settings.GOOGLE_GENAI_API_KEY,
    )


def get_openai_embeddings():
    return OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        model="text-embedding-3-large"
    )