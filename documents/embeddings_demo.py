# %%
from core.llm import get_openai_embeddings

embeddings = get_openai_embeddings()
# 对单个文本进行嵌入
text = "这是一个测试文本。"
embedding = embeddings.embed_query(text)
print(f"嵌入向量：{embedding}")
print(f"单个文本嵌入维度：{len(embedding)}")
