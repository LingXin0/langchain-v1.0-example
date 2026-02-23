from langchain.agents.middleware import dynamic_prompt, ModelRequest

from core.llm import get_model_anthropic
from langchain.agents import create_agent
from typing import TypedDict

# 系统提示词 System prompt
'''
Agent_v1 = create_agent(
    system_prompt="你是一个AI助手，可以回答任何问题"
)
'''


# -------------------------------------------------------------------------------- #


#  动态系统提示词 (Dynamic system prompt)
#      对于需要根据运行时上下文或agent状态修改系统提示符的更高级用例，可以使用中间件。
#      @dynamic_prompt装饰器创建中间件，根据模型请求动态生成系统提示：

# 自定义context格式
class Context(TypedDict):
    user_role: str


@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """根据用户角色来生成系统提示词."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "你是一个有用的助手."

    if user_role == "expert":
        return f"{base_prompt} 提供详细的技术回应."
    elif user_role == "beginner":
        return f"{base_prompt} 简单地解释概念，避免行话."

    return base_prompt


agent = create_agent(
    model=get_model_anthropic(),
    tools=[],
    middleware=[user_role_prompt],
    context_schema=Context,  # 在context中提供user_role
    # debug=True  # 调试模式
)

# 基于context中user_role的改变，系统提示词会动态变化
response = agent.invoke(
    {"messages": [{"role": "user", "content": "解释一下什么是机器学习"}]},
    # context={"user_role": "beginner"},  # 初学者
    context={"user_role": "expert"}  # 专家
)
print(response)
