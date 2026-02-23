from langchain_chroma import Chroma
from core.llm import get_openai_embeddings
from langchain_core.documents import Document

# 嵌入模型：
embeddings = get_openai_embeddings()

# 评分方式
score_measures = [
    "default",  # default = 'l2'
    "cosine",  # 余弦相似度， 1-cos(角度) 0-2
    "l2",  # 欧氏距离，
    "ip"  # 点积， 和 cosine 接近
]

# 创建向量库
db = Chroma(
    collection_name="collection_name",
    embedding_function=embeddings,
    persist_directory="./chroma_db1",
    collection_metadata={"hnsw:space": 'cosine'}
)

# 提供文档
documents = [
    Document(page_content="这个苹果手机很好用"),
    Document(page_content="我国山东地区盛产苹果"),
]

# 添加文档
ids = db.add_documents(documents)
print(ids)
print('-' * 100)

# 检索
results = db.similarity_search_with_score("我想买个手机")
for doc, score in results:
    print(doc.page_content, end='\t')
    print(f"score: {score}")



