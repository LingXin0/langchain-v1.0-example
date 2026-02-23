# %%
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langgraph.checkpoint.memory import InMemorySaver
from core.llm import get_model_anthropic, get_model_openai

#  动态模型 Dynamic model
#   动态模型是在运行时根据当前状态和上下文选择的。这使得复杂的路由逻辑和成本优化成为可能。
#   要使用动态模型，使用@wrap_model_call装饰器创建中间件，在请求中修改模型：
basic_model = get_model_anthropic()
advanced_model = get_model_openai()


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """根据会话复杂度选择模型."""
    message_count = len(request.state["messages"])
    # print(request.state)
    print("message_count:", message_count)
    # 根据会话长度，修改模型
    if message_count > 5:
        # 使用高级模式进行更长的对话
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))


# 设置记忆：支持多轮会话
checkpointer = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}  # 会话的唯一标识符。

# 创建agent
agent = create_agent(
    model=basic_model,  # 默认模型
    tools=[],
    middleware=[dynamic_model_selection],
    checkpointer=checkpointer,
)

# 执行agent
for _ in range(5):
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "2026年元宵节是几号"}]},
        config=config)
    print(response)