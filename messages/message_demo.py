import time

from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import ToolMessage

from core.llm import get_model_anthropic

# 在 LangChain 中，消息是模型的基本上下文单元。它们代表模型的输入和输出，携带与 LLM 交互时表示对话状态所需的内容和元数据。
# 消息是包含以下内容的对象：
#   角色 —— 标识消息类型（例如 system，user）
#   内容 —— 指消息的实际内容（例如文本、图像、音频、文档等）。
#   元数据 —— 可选字段，例如响应信息、消息 ID 和令牌使用情况
# LangChain 提供了一种适用于所有模型提供程序的标准消息类型，确保无论调用哪个模型，行为都保持一致。

# SystemMessage, HumanMessage, AIMessage
'''
messages = [
    SystemMessage("你是个诗歌专家"),
    HumanMessage("写一篇关于春天的七言绝句"),
    AIMessage("桃花盛开...")
]'''

# 字典格式
'''
messages = [
    {"role": "system", "content": "你是个诗歌专家"},
    {"role": "user", "content": "写一篇关于春天的七言绝句"},
    {"role": "assistant", "content": "桃花盛开..."}
]'''

# 消息元数据
#    该name字段的行为因提供商而异 —— 有些提供商将其用于用户识别，有些则忽略它。要查看具体情况

messages = [
    HumanMessage(
        content="Hello!",
        name="alice",  # 可选: 用来区别不同用户
        id="msg_123",  # 可选: 用于跟踪的唯一标识符
    )]

# 聊天模型
response = get_model_anthropic().invoke(messages)
print(response)
print('-' * 100)


# -------------------------------------------------------------------------------- #
'''
# 定义工具并调用
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    print('tool调用：get_weather')
    return f'今天天气真好，在 {location} 这个魅力的城市！'

# model_with_tools = qwen.bind_tools([get_weather])
# response = model_with_tools.invoke("长沙天气怎么样?")
# 
# for tool_call in response.tool_calls:
#     print(f"Tool: {tool_call['name']}")  # Tool: get_weather 工具名（即函数名）
#     print(f"Args: {tool_call['args']}")  # Args: {'location': '长沙'} 传给工具的参数
#     print(f"ID: {tool_call['id']}")  # ID: call_e68e9d4bd0e64d8a8c54ba 唯一标识符，用于后续匹配工具执行结果
# 
# print(response)
# print('-' * 100)

# -------------------------------------------------------------------------------- #

# 工具消息
#   对于支持工具调用的模型，AI 消息可以包含工具调用。工具消息用于将单次工具执行的结果传递回模型。
#   工具可以直接生成ToolMessage对象。
ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "长沙"},
        "id": "call_123"
    }]
)

# ToolMessage，执行工具后，作为工具执行结果 传递回模型

tool_message = ToolMessage(
    content="天气晴朗, 12°",
    tool_call_id="call_123"  # 必须匹配 call ID
)

# 对话
messages = [
    HumanMessage("长沙天气怎么样"),
    ai_message,  # AI模型消息
    tool_message,  # 工具消息
]
response = qwen.invoke(messages)
print(response.content_blocks)
print(response)

'''