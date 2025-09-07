import os, math, datetime, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.docstore.in_memory import InMemoryDocstore
from config import init_LLM, init_embedding

llm = init_LLM()
embed = init_embedding()

# =============================================================================
# 主題：每天喝一杯咖啡是否能延長壽命？
# 2023：肯定；2024：調整混雜因子後效果不顯著；2025：新方法指部份族群可能有風險→建議不以延壽為目的
# 干擾範例：優格與腸道健康（跟「延壽」不直接相關）
# =============================================================================

def build_articles() -> List[Dict[str, Any]]:
    return [
        {
            "id": "A_2023",
            "title": "研究：每日一杯咖啡或可延長壽命",
            "date": "2023-04-12",
            "summary": "觀察性研究指出，適量咖啡與較低的全因死亡率相關；研究者推測抗氧化與代謝效應可能是原因。",
            "sections": [
                {
                    "text": "一項涵蓋多國的觀察性研究顯示，每天喝一杯咖啡的人，平均全因死亡率較未飲用者低約5%。此關聯在不同年齡層與性別中皆可見。",
                    "summary": "觀察到咖啡與較低死亡率關聯（約5%）。"
                },
                {
                    "text": "研究者推測，咖啡中的多酚與咖啡因可能透過抗氧化與改善代謝，帶來長期健康效益。不過研究仍呼籲以適量為原則。",
                    "summary": "可能機制：多酚、咖啡因帶來抗氧化與代謝效益。"
                },
                {
                    "text": "不過，研究主要採問卷與回溯資料，仍可能存在生活型態差異等混雜因子；研究者建議未來進一步驗證。",
                    "summary": "研究限制：觀察性、可能有混雜因子。"
                }
            ]
        },
        {
            "id": "B_2024",
            "title": "新分析：調整生活型態後，咖啡與壽命關聯不再明顯",
            "date": "2024-08-20",
            "summary": "多項研究的整合分析顯示，當納入吸菸、運動、飲食等混雜因子後，咖啡與較長壽命的關聯顯著下降，接近於無效應。",
            "sections": [
                {
                    "text": "一項整合過去十年的多研究分析指出，若嚴格調整吸菸、運動與飲食品質等變項，咖啡攝取與壽命延長的關聯性降至不顯著。",
                    "summary": "嚴格調整混雜因子後，不再顯著。"
                },
                {
                    "text": "研究者指出，早期研究可能高估咖啡的保護效果，部分原因在於喝咖啡者同時擁有較健康的生活型態。",
                    "summary": "早期研究可能高估效果，健康人偏差所致。"
                },
                {
                    "text": "該分析強調，若以延長壽命為目的而刻意增加咖啡攝取，並無足夠證據支持；維持規律運動與充足睡眠對壽命的影響更確定。",
                    "summary": "不建議為延壽而刻意多喝；運動睡眠更重要。"
                }
            ]
        },
        {
            "id": "C_2025",
            "title": "最新方法：特定族群過量咖啡或增心血管風險",
            "date": "2025-06-01",
            "summary": "以類隨機方法重新分析資料，發現某些遺傳型態或心血管高風險族群在高咖啡攝取下，事件風險可能上升；不建議以延壽為目的飲用咖啡。",
            "sections": [
                {
                    "text": "研究使用遺傳工具變數進行因果推斷，發現高咖啡攝取對特定遺傳型態的受試者，心血管事件風險增加。",
                    "summary": "因果工具變數：某些族群高咖啡→風險上升。"
                },
                {
                    "text": "整體族群平均效果接近中性，但族群異質性明顯：部分人無明顯影響，部分人風險上升，提示需個別化建議。",
                    "summary": "平均中性，族群差異大。"
                },
                {
                    "text": "結論指出，若以延長壽命為目標，增加咖啡攝取不具一致性證據，反而應優先管理睡眠、運動與慢病風險。",
                    "summary": "不建議為延壽多喝咖啡；改做基本健康行為。"
                }
            ]
        },
        {
            "id": "D_2024_misc",
            "title": "優格有助腸道菌相多樣性，與壽命未見直接關聯",
            "date": "2024-03-02",
            "summary": "小型實驗指出優格可改善腸道菌相與部分代謝指標，但尚未建立與壽命的直接因果關係。",
            "sections": [
                {
                    "text": "受試者每日攝取優格四週，觀察到腸道菌相多樣性提升，並伴隨部分發炎指標下降。",
                    "summary": "優格改善菌相與發炎指標。"
                },
                {
                    "text": "研究規模有限、追蹤時間短，且未觀察到與壽命指標的直接連結。",
                    "summary": "規模小、追蹤短，與壽命未建立關聯。"
                },
                {
                    "text": "作者提醒讀者，仍需更長期的大型研究。",
                    "summary": "需要更長期研究。"
                }
            ]
        },
    ]

articles = build_articles()


import json
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

def infer_stance(t):
    p = ["延長壽命","延壽","有助","較低死亡率"]
    c = ["不再顯著","無足夠證據","風險上升","不建議","沒有直接關聯","無直接因果"]
    if any(x in t for x in c): return "contra"
    if any(x in t for x in p): return "pro"
    return "neutral"

def to_documents(arts):
    docs = []
    for a in arts:
        docs.append(Document(page_content=a["summary"], metadata={"doc_id":a["id"],"title":a["title"],"date":a["date"],"section":"摘要","type":"doc_summary","stance":infer_stance(a["summary"])}))
        for i,s in enumerate(a["sections"],start=1):
            docs.append(Document(page_content=s["text"], metadata={"doc_id":a["id"],"title":a["title"],"date":a["date"],"section":f"{i}","type":"section","stance":infer_stance(s["text"])}))
            docs.append(Document(page_content=s["summary"], metadata={"doc_id":a["id"],"title":a["title"],"date":a["date"],"section":f"{i}","type":"evidence","stance":infer_stance(s["summary"])}))
    return docs

def rrf_fuse(rank_lists, k=24, k_rrf=60):
    scores = {}
    keep = {}
    for lst in rank_lists:
        for r,d in enumerate(lst):
            key = f'{d.metadata.get("doc_id")}||{d.metadata.get("section")}||{d.page_content[:40]}'
            scores[key] = scores.get(key,0.0)+1.0/(k_rrf+r+1)
            keep[key]=d
    ranked = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    return [keep[k] for k,_ in ranked[:k]]

def diversify_by_doc(docs, k=8):
    out = []
    seen = set()
    for d in docs:
        did = d.metadata.get("doc_id")
        if did in seen: continue
        seen.add(did)
        out.append(d)
        if len(out)>=k: break
    if len(out)<k:
        for d in docs:
            if d not in out:
                out.append(d)
                if len(out)>=k: break
    return out[:k]

def sort_by_recency(docs):
    return sorted(docs, key=lambda d:d.metadata.get("date","1970-01-01"), reverse=True)

def generate_queries(llm, q):
    tpl = ChatPromptTemplate.from_messages([("system","你是檢索助理，輸出4行不同措辭的查詢，不要解釋"),("human","問題：{q}")])
    resp = llm.invoke(tpl.format_messages(q=q)).content.strip().splitlines()
    cand = [q]+[x.strip(" -•").strip() for x in resp if x.strip()]
    uniq=[]
    s=set()
    for v in cand:
        if v not in s:
            uniq.append(v); s.add(v)
    return uniq[:4]

def citations_str(dlist, n=6):
    out=[]
    for d in dlist[:n]:
        m=d.metadata
        out.append(f'[{m.get("title")} {m.get("date")} §{m.get("section")}] {d.page_content}')
    return out

def build_basic_chain(llm):
    doc_prompt = PromptTemplate.from_template("【{title} {date} §{section}】\n{page_content}")
    prompt = ChatPromptTemplate.from_messages([("system","僅依提供內容回答，若不確定則說不確定，列出來源"),("human","問題：{question}\n\n{context}")])
    return create_stuff_documents_chain(llm, prompt, document_prompt=doc_prompt)

def build_advanced_chain(llm):
    doc_prompt = PromptTemplate.from_template("【{title} {date} §{section}】\n{page_content}")
    prompt = ChatPromptTemplate.from_messages([("system","優先採用日期較新內容；若內容互相矛盾，並列新舊說法與日期；僅依提供內容回答並列出來源"),("human","問題：{question}\n\n{context}")])
    return create_stuff_documents_chain(llm, prompt, document_prompt=doc_prompt)

docs = to_documents(articles)
bm25 = BM25Retriever.from_documents(docs)
bm25.k = 20
faiss = FAISS.from_documents(docs, embed)
vretr = faiss.as_retriever(search_kwargs={"k":20})

basic_chain = build_basic_chain(llm)
adv_chain = build_advanced_chain(llm)

def ask_basic(q):
    hits = bm25.get_relevant_documents(q)[:6]
    ans = basic_chain.invoke({"question":q,"context":hits})
    return ans, citations_str(hits,6)

def ask_advanced(q):
    qs = generate_queries(llm, q)
    pools = []
    for qq in qs:
        pools.append(bm25.get_relevant_documents(qq))
        pools.append(vretr.get_relevant_documents(qq))
    fused = rrf_fuse(pools, k=24)
    diverse = diversify_by_doc(fused, k=8)
    ranked = sort_by_recency(diverse)
    ans = adv_chain.invoke({"question":q,"context":ranked})
    return ans, citations_str(ranked,6), qs

q1 = "延壽的最新共識是什麼？是否推翻早期的咖啡說法？請附日期與引用"
b1, b1_c = ask_basic(q1)
a1, a1_c, a1_q = ask_advanced(q1)
print("=== 基本版 ===")
print(b1)
print("引用：")
for c in b1_c: print("-", c)
print("\n=== 進階版 ===")
print(a1)
print("引用：")
for c in a1_c: print("-", c)
print("Multi-Query：", a1_q)

q2 = "優格是否能直接延長壽命？"
b2, b2_c = ask_basic(q2)
a2, a2_c, a2_q = ask_advanced(q2)
print("\n=== 基本版（干擾題） ===")
print(b2)
print("引用：")
for c in b2_c: print("-", c)
print("\n=== 進階版（干擾題） ===")
print(a2)
print("引用：")
for c in a2_c: print("-", c)
print("Multi-Query：", a2_q)
