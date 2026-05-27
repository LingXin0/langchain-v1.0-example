# %%
from deepagents import create_deep_agent
from core.llm import get_model_anthropic, get_daytona_config
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from urllib.request import urlopen
from tools.websearch import internet_search, get_weather
# from langchain_modal import ModalSandbox
from daytona import Daytona
from langchain_daytona import DaytonaSandbox
from langgraph.checkpoint.memory import MemorySaver
from tools.deepagent_tools import delete_file, read_file, send_email
from pydantic import BaseModel, Field

# %%


agent = create_deep_agent(
    model=get_model_anthropic(),
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
res = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
for item in res.items():
    print(item)

# %%  Tools | Model | System Prompt

# System prompt to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

agent = create_deep_agent(
    model=get_model_anthropic(),
    tools=[internet_search],
    system_prompt=research_instructions,
    debug=True
)
result = agent.invoke({"messages": [{"role": "user", "content": "What is English grammar?"}]})

# Print the agent's response
print(result["messages"][-1].content)

# %% Subagents
research_subagent = {
    "name": "research-agent",
    "description": "Used to research more in depth questions",
    "system_prompt": "You are a great researcher",
    "tools": [internet_search],
    "model": get_model_anthropic(),  # Optional override, defaults to main agent model
}
subagents = [research_subagent]

agent = create_deep_agent(
    model=get_model_anthropic(),
    subagents=subagents,
    debug=True

)
result = agent.invoke({"messages": [{"role": "user", "content": "What is English grammar?"}]})

# Print the agent's response
print(result["messages"][-1].content)

# %%  StateBackend | FilesystemBackend | LocalShellBackend | StoreBackend | CompositeBackend
agent = create_deep_agent(
    model=get_model_anthropic(),
    backend=(lambda rt: StateBackend(rt))  # Note that the tools access State through the runtime.state
    # backend=FilesystemBackend(root_dir=".", virtual_mode=True)
    # backend=LocalShellBackend(root_dir=".", env={"PATH": "/usr/bin:/bin"})
)
# ~StoreBackend is taken for saving data in the cloud~
# agent = create_deep_agent(
#     backend=(lambda rt: StoreBackend(rt)),
#     store=InMemoryStore()  # Good for local dev; omit for LangSmith Deployment
# )
# ~CompositeBackend is taken for Handling complex tasks~
# composite_backend = lambda rt: CompositeBackend(
#     default=StateBackend(rt),
#     routes={
#         "/memories/": StoreBackend(rt),
#     }
# )
#
# agent = create_deep_agent(
#     backend=composite_backend,
#     store=InMemoryStore()  # Store passed to create_deep_agent, not backend
# )
result = agent.invoke({"messages": [{"role": "user", "content": "What is English grammar?"}]})

# Print the agent's response
print(result["messages"][-1].content)

# %% Sandboxes  Daytona

sandbox = Daytona(get_daytona_config()).create()
backend = DaytonaSandbox(sandbox=sandbox)

agent = create_deep_agent(
    model=get_model_anthropic(),
    system_prompt="You are a Python coding assistant with sandbox access.",
    backend=backend
)

try:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Create a small Python package and run pytest",
                }
            ]
        }
    )
    print(result["messages"][-1].content)
finally:
    sandbox.stop()

# %%  modal
import modal
from langchain_anthropic import ChatAnthropic

from deepagents import create_deep_agent
from langchain_modal import ModalSandbox

app = modal.App.lookup("your-app")
modal_sandbox = modal.Sandbox.create(app=app)
backend = ModalSandbox(sandbox=modal_sandbox)

agent = create_deep_agent(
    model=ChatAnthropic(model="claude-sonnet-4-20250514"),
    system_prompt="You are a Python coding assistant with sandbox access.",
    backend=backend,
)
try:
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Create a small Python package and run pytest",
                }
            ]
        }
    )
finally:
    modal_sandbox.terminate()

# %% Human-in-the-loop
# Checkpointer is REQUIRED for human-in-the-loop
checkpointer = MemorySaver()

agent = create_deep_agent(
    model=get_model_anthropic(),
    tools=[delete_file, read_file, send_email],
    interrupt_on={
        "delete_file": True,  # Default: approve, edit, reject
        "read_file": False,  # No interrupts needed
        "send_email": {"allowed_decisions": ["approve", "reject"]},  # No editing
    },
    checkpointer=checkpointer  # Required!
)
# %% Skills https://github.com/langchain-ai/deepagentsjs/tree/main/examples/skills
checkpointer = MemorySaver()

skill_url = "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/libs/cli/examples/skills/langgraph-docs/SKILL.md"
with urlopen(skill_url) as response:
    skill_content = response.read().decode('utf-8')

skills_files = {
    "/skills/langgraph-docs/SKILL.md": create_file_data(skill_content)
}

agent = create_deep_agent(
    model=get_model_anthropic(),
    skills=["./skills/"],
    checkpointer=checkpointer,
    debug=True,
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "What is langgraph?",
            }
        ],
        # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
        "files": skills_files
    },
    config={"configurable": {"thread_id": "12345"}},
)
print(result["messages"][-1].content)

# %% Skills Memory
with urlopen(
        "https://raw.githubusercontent.com/langchain-ai/deepagents/refs/heads/main/examples/text-to-sql-agent/AGENTS.md") as response:
    agents_md = response.read().decode("utf-8")
checkpointer = MemorySaver()


class FinalAnswer(BaseModel):
    text: str


agent = create_deep_agent(
    model=get_model_anthropic(),
    memory=[
        "/AGENTS.md"
    ],
    checkpointer=checkpointer,
    response_format=FinalAnswer,
    debug=True,
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Please tell me what's in your memory files.",
            }
        ],
        # Seed the default StateBackend's in-state filesystem (virtual paths must start with "/").
        "files": {"/AGENTS.md": create_file_data(agents_md)},
    },
    config={"configurable": {"thread_id": "123456"}},
)
print(result["structured_response"].text)


# %% Structured-output  https://docs.langchain.com/oss/python/langchain/structured-output
class WeatherReport(BaseModel):
    """A structured weather report with current conditions and forecast."""
    location: str = Field(description="The location for this weather report")
    temperature: float = Field(description="Current temperature in Celsius")
    condition: str = Field(description="Current weather condition (e.g., sunny, cloudy, rainy)")
    humidity: int = Field(description="Humidity percentage")
    wind_speed: float = Field(description="Wind speed in km/h")
    forecast: str = Field(description="Brief forecast for the next 24 hours")


agent = create_deep_agent(
    model=get_model_anthropic(),
    tools=[internet_search],
    response_format=WeatherReport
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "What's the weather like in Shenzhen, China?"
    }]
})

print(result["structured_response"])
# location='San Francisco, California' temperature=18.3 condition='Sunny' humidity=48 wind_speed=7.6 forecast='Pleasant sunny conditions expected to continue with temperatures around 64°F (18°C) during the day, dropping to around 52°F (11°C) at night. Clear skies with minimal precipitation expected.'
