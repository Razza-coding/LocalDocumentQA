import faiss
from faiss import IndexFlat
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import json

# 基本設定
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=encode_kwargs
)
dimension  = len(embeddings.embed_query("Hello World")) # 使用 Embedding 向量長度當作 VecDB 維度

# 建立 DataBase
index = faiss.IndexFlatL2(dimension) # 選擇 Index 類型，使用 L2 距離

# 使用 Hugging Face 包裝 VecDB
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

'''
備註：原生 faiss 不包含文件儲存與文件對應，但 Hugging Face 版有封裝高階功能
但是這不代表原本 faiss 功能可以不學習，從 HF FAISS 抽取 index 使用時就是用 faiss 函式操作
HF FAISS 內部包含(以下假設文件是目標資料)：
1. embedding : 文件轉向量
2. index     : 向量搜索，輸入搜索向量 獲得 UUID (來自 hashing 或 使用者定義)
3. storage   : 文件儲存倉庫，UUID 對應 文件
搜索流程：
1. query text -> embedding -> query vector
2. query vector -> index similarity search -> top K similar item -> top k UUID
3. UUID -> fetch from storge -> get document in database
加入資料流程
1. input doc -> text splitting/preprocessing -> embedding -> input vector
2. input vector -> add in index -> generate UUID
3. UUID -> create mapping in storage (might be hashing) -> add UUID : input doc mapping
'''

# 輸入資料
texts    = [
    "small gray mouse carry a big slice of cheese",
    "round orange cat eating pasta",
    "black dog running around the park with tennis ball"
    ]
metadata = [{"baz" : "bar"}, {"bar" : "baz"}, {"tgt": "del"}]
documents = [Document(page_content=texts[i], metadata=metadata[i]) for i in range(len(texts))]

vector_store.add_documents(documents=documents)

# 加入一個新資料
new_text = "young owl that hunts at night"
new_data = Document(page_content=new_text)
vector_store.add_documents([new_data])

# 執行搜尋
querys = [
    texts[0], 
    "Steve looking for an orange animal", 
    "after a while, it is sleeping right now", 
    "it moves really fast", 
    "slowly but surely, eventually finished"
    ]

for q in querys:
    print(f"Queary: {q}")
    # 執行搜尋
    result = vector_store.similarity_search_with_score(query=q, k=5)

    for doc, score in result:
        print(f"\tId: {doc.id :<3} Score: {score :<4f} Text: {doc.page_content :<60} MetaData: {str(doc.metadata) :<}")

# 存檔
save_root = "LocalVecDB\\{0}"
# faiss 存檔，無對應文件資料
faiss.write_index(index, save_root.format("faiss_temp.index"))
# Hugging face faiss 存檔，index 與 pkl 檔一起儲存
vector_store.save_local(save_root.format(""), "HF_faiss_temp.index")
# 自行完成的存檔方式，資料可閱讀
def dump_vecDB(vector_database: FAISS) -> None:
    data = []
    index = vector_database.index
    v = index.reconstruct_n(0, index.ntotal)
    for i in range(index.ntotal):
        doc_vector = v[i]
        doc_id   = vector_store.index_to_docstore_id[i]
        doc_text = vector_store.docstore.search(doc_id).page_content
        data.append({
            "id"     : str(doc_id),
            "vector" : str(doc_vector),
            "text"   : str(doc_text)
        })
    with open(save_root.format("HF_faiss_vector_database.json", encoding="Big5"), mode="w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return
dump_vecDB(vector_store)

'''
Note
 - Score 越低代表相似度越高
 - 相似度除了代表與句子本身相近，也可能代表接在後面的句子
 - id 未定義時使用 UUID，可用 int 替代
 - 前 3 組高機率包含最佳匹配，但第一名不一定是最佳匹配
 - 更換 Embedding 可微調搜尋結果，微量增強搜索品質
'''

# 視覺化
def visualize_vector_database(vector_database: IndexFlat) -> list:
    # 使用全部
    raw_vector = vector_database.reconstruct_n(0, vector_database.ntotal)
    perplexity = min(raw_vector.shape[0] - 1, 30)
    reduction = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    vector_2d = reduction.fit_transform(raw_vector)
    return vector_2d

vec_2d = visualize_vector_database(index)
plt.scatter(vec_2d[:, 0], vec_2d[:, 1], s=5, alpha=0.6)
plt.title("FAISS vector visualize")
plt.show()