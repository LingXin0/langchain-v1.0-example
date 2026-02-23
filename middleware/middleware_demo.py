# %%
# from langchain.agents import create_agent
# from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware
# from langchain_core.tools import tool

from core.llm import get_model_anthropic, get_model_openai

# Middleware 中间件

# 添加中间件的方式：在create_agent
'''
Agent_v1 = create_agent(
    model=qwen,
    tools=[],
    middleware=[SummarizationMiddleware(), HumanInTheLoopMiddleware()],
)
'''

# 内置中间件
#   LangChain 为常见用例提供预构建的中间件：

# SummarizationMiddleware : 总结摘要的中间件
#       当接近会话次数上限时，自动汇总对话历史记录。
#  非常适合：
#     - 持续时间过长的对话超出了上下文窗口。
#     - 多轮对话，历史悠久
#     - 在需要保留完整对话上下文的应用中
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model=get_model_anthropic(),
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model=get_model_openai(),
            # max_tokens_before_summary=4000,  # 4000个token 会触发 摘要总结  # V1.0写法
            trigger=('tokens', 4000),  # V1.1写法
            # messages_to_keep=20,  # 在总结后保留最后20条消息  # V1.0写法
            keep=('messages', 20),  # V1.1写法
            summary_prompt=" 可以自定义进行摘要的提示词...",  # 可选
        ),
    ],
)