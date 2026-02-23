# %%
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_model_anthropic

#  LCEL: LangChain表达式（LangChain Expression Language）

# 提示词
prompt = ChatPromptTemplate([
    ("system", "把用户输入的中文翻译成{language}"),
    ("user", "{text}"),
])

# 输出解析器
parser = StrOutputParser()


# 链（简单）
_chain = prompt | get_model_anthropic() | parser

# 调用执行链
for e in _chain.stream({"language": "英文", "text": "Jeff喜欢打篮球"}):
    print(e)



