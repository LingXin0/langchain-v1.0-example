from langchain.agents import create_agent
from langchain.tools import tool

from core.llm import get_model_anthropic


# 基本工具定义
@tool
def search_database(query: str, limit: int = 10) -> str:
    """在客户数据库中搜索匹配查询的记录

    Args:
        query: 要查找的搜索词
        limit: 返回的最大结果数
    """
    return f"这个查询:'{query}', 找到了 {limit} 条结果！"


# 自定义工具名称
@tool("web_search")
def search(query: str) -> str:
    """在网上搜索信息."""
    return f"{query}的查询结果！"


print(search.name)


# 自定义工具描述
@tool("calculator", description="执行算术计算。用它来解决任何数学问题.")
def calc(expression: str) -> str:
    """计算数学表达式的值."""
    return str(eval(expression))


# 高级模式定义
#   使用 Pydantic 模型或 JSON 模式定义复杂输入
from pydantic import BaseModel, Field
from typing import Literal


class WeatherInput(BaseModel):
    """天气查询的输入."""
    location: str = Field(description="城市名称或坐标")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="温度单位"
    )
    include_forecast: bool = Field(
        default=False,
        description="包括五天天气预报"
    )


@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """获取当前天气和可选的天气预报."""
    temp = 22 if units == "celsius" else 72
    result = f"城市 {location} 的当前温度是: {temp} ，单位： {units[0].upper()}"
    if include_forecast:
        result += "\n未来5天: Sunny"
    return result


# JSON 模式
weather_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string"},
        "include_forecast": {"type": "boolean"}
    },
    "required": ["location", "units", "include_forecast"]
}


@tool(args_schema=weather_schema)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """获取当前天气和可选的天气预报."""
    temp = 22 if units == "celsius" else 72
    result = f"城市 {location} 的当前温度是: {temp} ，单位： {units[0].upper()}"
    if include_forecast:
        result += "\n未来5天: Sunny"
    return result


# agent调用工具
agent = create_agent(get_model_anthropic(),
                     tools=[search, get_weather])
response = agent.invoke({"messages": [{"role": "user", "content": "长沙明天天气怎么样?"}]})
print(response)
