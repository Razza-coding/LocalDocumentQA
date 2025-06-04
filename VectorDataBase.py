import faiss
from faiss import IndexFlat
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 基本設定
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs=embedding_model_kwargs,
    encode_kwargs=encode_kwargs
)
dimension  = len(embeddings.embed_query("Hello World")) # 使用 Embedding 向量長度當作 VB 維度

# 建立 DataBase
index = faiss.IndexFlatL2(dimension) # 選擇 Index 類型，使用 L2 距離

# 使用 Hugging Face 包裝 VB
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

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
save_root = "LocalVB\\{0}"
faiss.write_index(index, save_root.format("faiss_temp.index")) # faiss 存檔
vector_store.save_local(save_root.format(""), "HF_faiss_temp.index")


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