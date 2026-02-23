from langchain_core.runnables import RunnableLambda, chain, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter

from core.llm import get_model_anthropic
from langchain_core.output_parsers import StrOutputParser

# 高级链：RunnableLambda
# itemgetter
'''
d = {"foo": "abc",   "bar": "abcd"}
ret = itemgetter("foo")(d)
print(ret)  # "abc"
exit()
'''

# 定义提示模版
template = ChatPromptTemplate.from_template("{a} + {b}是多少？")


# 获得字符串的长度
def length(t):
    return len(t)


# 将两个字符串长度的数量相乘
def mul(t1, t2):
    return len(t1) * len(t2)


# @chain是RunnableLambda的另一种写法: 把函数转换为与LCEL兼容的组件
@chain
def mul_length(d):
    return mul(d["t1"], d["t2"])


chain1 = template | get_model_anthropic()
chain2 = (
        {
            "a": itemgetter("name") | RunnableLambda(length),  # a=6
            "b": {"t1": itemgetter("name"), "t2": itemgetter("sex")} | mul_length,  # b=24
        }
        | chain1
        | StrOutputParser()
)
print(chain2.invoke({"name": "Darius", "sex": "male"}))
print('-' * 100)
