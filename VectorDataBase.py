import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# 基本設定
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=encode_kwargs
)
dimension  = len(embeddings.embed_query("Hello World")) # 使用 Embedding 向量長度當作 VB 維度

# 建立 DataBase
index = faiss.IndexFlatL2(dimension) # 建立資料庫，使用 L2 距離
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

texts    = [
    "small gray mouse carry a big slice of cheese",
    "round orange cat eating pasta",
    "black dog running around the park with tennis ball"
    ]
metadata = [{"baz" : "bar"}, {"bar" : "baz"}, {"tgt": "del"}]
documents = [Document(page_content=texts[i], metadata=metadata[i]) for i in range(len(texts))]
ids = [str(id+1) for id in range(len(documents))]
vector_store.add_documents(documents=documents, ids=ids)

# 加入新資料
new_data = Document(page_content="young owl that hunts at night")
new_id = 99
vector_store.add_documents([new_data], ids=[new_id])

# 搜尋
querys = [
    texts[0], 
    "Steve looking for an orange animal", 
    "after a while, it is sleeping right now", 
    "it moves really fast", 
    "slowly but surely, eventually finished"
    ]
for q in querys:
    print(f"Queary: {q}")
    result = vector_store.similarity_search_with_score(query=q, k=5)
    for doc, score in result:
        print(f"\tId: {doc.id :<3} Score: {score :<4f} Text: {doc.page_content :<60} MetaData: {str(doc.metadata) :<}")

# VB 存檔
faiss.write_index(index, "LocalVB\\temp.indx")
# 取出 index 中的所有向量
vectors = index.reconstruct_n(0, index.ntotal)
# 寫入 txt
np.savetxt("LocalVB\\faiss_vectors.txt", vectors, fmt="%.6f")


'''
Note
 - Score 越低代表相似度越高
 - 相似度除了代表與句子本身相近，也可能代表接在後面的句子
 - id 未定義時會自己補上，但不是 int 形式記錄
 - 前 3 組高機率包含最佳匹配，但第一名不一定是最佳匹配
 - 更換 Embedding 可微調搜尋結果，微量增強搜索品質
'''