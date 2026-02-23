# %%
from langchain.tools import tool
from langchain.agents.middleware import wrap_tool_call, AgentMiddleware
from deepagents import create_deep_agent
from core.llm import get_model_anthropic


@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."


call_count = [0]  # Use list to allow modification in nested function


# Update graph state instead
class CustomMiddleware(AgentMiddleware):
    def __init__(self):
        pass

    def before_agent(self, state, runtime):
        return {"x": state.get("x", 0) + 1}


@wrap_tool_call
def log_tool_calls(request, handler):
    """Intercept and log every tool call - demonstrates cross-cutting concern."""
    call_count[0] += 1
    tool_name = request.name if hasattr(request, 'name') else str(request)

    print(f"[Middleware] Tool call #{call_count[0]}: {tool_name}")
    print(f"[Middleware] Arguments: {request.args if hasattr(request, 'args') else 'N/A'}")

    # Execute the tool call
    result = handler(request)

    # Log the result
    print(f"[Middleware] Tool call #{call_count[0]} completed")

    return result


agent = create_deep_agent(
    model=get_model_anthropic(),
    tools=[get_weather],
    middleware=[log_tool_calls],
    debug=True
)
result = agent.invoke({"messages": [{"role": "user", "content": "深圳的天气怎么样？"}]})

# Print the agent's response
print(result["messages"][-1].content)