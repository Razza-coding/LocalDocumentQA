# run_tests.py
# 文字類 Benchmark 自動下載與範例測試流程 (方案 A)

from datasets import load_dataset
from pathlib import Path
import random
import os
import json

# 設定儲存資料夾
LOCAL_CACHE = os.path.abspath("./BanchMark")
LOCAL_HUB_CACHE = os.path.abspath("./BanchMark/Hub")
DATASET_DIR = Path("datasets")
DATASET_DIR.mkdir(exist_ok=True)
os.environ['HF_HOME'] = LOCAL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = LOCAL_HUB_CACHE

# Benchmark 分類與名稱
benchmarks = {
    "資訊抽取": [
        {"source" : "HF", "name" : "google/boolq",      "split" : "validation", "sub" : None},
        {"source" : "HF", "name" : "facebook/belebele", "split" : "test",       "sub" : "eng_Latn"}, 
        {"source" : "HF", "name" : "facebook/belebele", "split" : "test",       "sub" : "zho_Hant"}, 
        {"source" : "HF", "name" : "google/xquad",      "split" : "validation", "sub" : "xquad.en"},
        {"source" : "HF", "name" : "google/xquad",      "split" : "validation", "sub" : "xquad.zh"},
        ],
    "對話": [
        {"source" : "HF_file", "name" : "socialiqa-train-dev", "split" : "train", "sub" : "dev.jsonl"},
        {"source" : "HF", "name" : "google-research-datasets/natural_questions", "split" : "validation", "sub" : "default"}
        ],
    #"常識": ["mmlu", "global_mmlu_lite"],
    #"推理": ["mbpp", "gsm8k"],
    #"翻譯": ["wmt14"],  # wmt24++ 暫時以 wmt14 代表
}

def preview_dataset(source: str, name: str, split: str, sub):
    SPLIT_LINE = "-" * 30
    print(SPLIT_LINE)
    msg = ""
    msg += f"Load [{name}] - "

    try:
        ds = None
        if source == "HF":
            ds = load_dataset(name, sub, split=split, cache_dir=LOCAL_CACHE)
            msg += f"Samples [{len(ds)}]"
        elif source == "HF_file":
            file_path = os.path.join(os.path.abspath("./BanchMark"), name, sub)
            ds = load_dataset("json", data_files=file_path, split=split, cache_dir=LOCAL_CACHE)
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
    preview_amount = 1
    for category, dataset_info in benchmarks.items():
        for dataset in dataset_info:
            source = dataset['source']
            name   = dataset['name']
            sub    = dataset['sub']
            split    = dataset['split']
            ds = preview_dataset(source, name, split, sub)
            if ds is not None:
                print(ds[:preview_amount])

if __name__ == "__main__":
    main()
