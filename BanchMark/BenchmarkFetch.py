from pathlib import Path
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from LogWriter import LogWriter, remove_special_symbol

# Create custom download path
PATH_ROOT     = os.path.abspath("./BanchMark")
BASE_CACHE    = os.path.join(PATH_ROOT, "Dataset")
DATASET_CACHE = os.path.join(BASE_CACHE, "datasets")
HUB_CACHE     = os.path.join(BASE_CACHE, "hub")
Path(DATASET_CACHE).mkdir(parents=True, exist_ok=True)
Path(HUB_CACHE).mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = BASE_CACHE
os.environ["HF_DATASETS_CACHE"] = DATASET_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = HUB_CACHE
os.environ["HF_HUB_CACHE"] = HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = os.path.join(BASE_CACHE, "transformers")
os.environ["XDG_CACHE_HOME"] = BASE_CACHE

# import AFTER custom path is set
from datasets import load_dataset

# Benchmark 分類與名稱
benchmarks = {
    "資訊抽取": [
        #{"source" : "HF", "name" : "google/boolq",      "split" : "validation", "sub" : None},
        #{"source" : "HF", "name" : "facebook/belebele", "split" : "test",       "sub" : "eng_Latn"}, 
        #{"source" : "HF", "name" : "facebook/belebele", "split" : "test",       "sub" : "zho_Hant"}, 
        #{"source" : "HF", "name" : "google/xquad",      "split" : "validation", "sub" : "xquad.en"},
        #{"source" : "HF", "name" : "google/xquad",      "split" : "validation", "sub" : "xquad.zh"},
        ],
    "對話": [
        #{"source" : "HF_file", "name" : "socialiqa-train-dev", "split" : "train", "sub" : "dev.jsonl"},
        #{"source" : "HF", "name" : "google-research-datasets/natural_questions", "split" : "validation", "sub" : "default"}
        ],
    #"常識": ["mmlu", "global_mmlu_lite"],
    #"推理": ["mbpp", "gsm8k"],
    #"翻譯": ["wmt14"],  # wmt24++ 暫時以 wmt14 代表
    "RAG": [
        {"source" : "HF", "name" : "rag-datasets/rag-mini-wikipedia", "split" : "passages", "sub" : "text-corpus"}, # RAG 資料集
        {"source" : "HF", "name" : "rag-datasets/rag-mini-wikipedia", "split" : "test",    "sub" : "question-answer"}, # RAG 問答集
        {"source" : "HF", "name" : "corbt/all-recipes", "split" : "train", "sub" : None}, # 食譜
    ]
}

def preview_dataset(source: str, name: str, split: str, sub):
    SPLIT_LINE = "-" * 30
    print(SPLIT_LINE)
    msg = ""
    msg += f"Load [{name}] - "

    try:
        ds = None
        if source == "HF":
            ds = load_dataset(
                name,
                sub,
                split=split,
                cache_dir=DATASET_CACHE,
            )
            msg += f"Samples [{len(ds)}]"
        elif source == "HF_file":
            file_path = os.path.join(PATH_ROOT, name, sub)
            ds = load_dataset(
                "json",
                data_files=file_path,
                split=split,
                cache_dir=DATASET_CACHE,
            )
            msg += f"Samples [{len(ds)}]"
        else:
            msg += f" EMPTY "
        print(msg)
        return ds
    except Exception as e:
        msg += "FAILED"
        print(msg)
        return None

def main():
    preview_amount = 1000000
    for category, dataset_info in benchmarks.items():
        for dataset in dataset_info:
            source = dataset['source']
            name   = dataset['name']
            sub    = dataset['sub']
            split    = dataset['split']
            ds = preview_dataset(source, name, split, sub)
            log_name = remove_special_symbol(f"{name}-{sub}")
            log = LogWriter(log_name=f"{log_name}", log_folder_name="logs", root_folder=BASE_CACHE)
            log.clear()
            if ds is not None:
                for idx in range(min(len(ds), preview_amount)):
                    print(ds[idx])
                    log.write_log(ds[idx])
                

if __name__ == "__main__":
    main()
