# %%
from core.llm import get_model_anthropic
from langchain.chat_models import init_chat_model
from core.config import settings

# %%
res = get_model_anthropic().invoke("who are you!")
print(res.content)

# %%
client = init_chat_model(model="claude-sonnet-4-5", api_key=settings.ANTHROPIC_API_KEY)
response = client.invoke("who are you!")
print(response.content)

# %%
# model_anthropic = get_model_anthropic()
# model_anthropic.batch("who are you!")
