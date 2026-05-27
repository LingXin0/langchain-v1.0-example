from typing import List

from daytona import DaytonaConfig
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
        model_name="claude-sonnet-4-6",
        api_key=settings.ANTHROPIC_API_KEY,
        timeout=60.0,
        stop=stop_args,
        streaming=True,
    )


def get_model_google_genai():
    return ChatGoogleGenerativeAI(
        model_name="gemini-3.1-pro-preview",
        api_key=settings.GOOGLE_GENAI_API_KEY,
    )

def get_model_mimo(stop_args: List[str] = None):
    # MiMo 提供 OpenAI 兼容 API,直接复用 ChatOpenAI 指向其 base_url 即可
    return ChatOpenAI(
        model_name="mimo-v2.5-pro",          # 以平台实际模型名为准
        api_key=settings.MIMO_API_KEY,
        base_url="https://api.xiaomimimo.com/v1",  # 以平台文档实际地址为准
        timeout=60.0,
        stop=stop_args,
        streaming=True,
    )


def get_openai_embeddings():
    return OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY,
        model="text-embedding-3-large"
    )


def get_daytona_config(stop_args: List[str] = None):
    return DaytonaConfig(
        api_key=settings.DAYTONA_API_KEY
    )
