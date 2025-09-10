from pathlib import Path
import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from LogWriter import LogWriter, remove_special_symbol
from HFCacheSetting import set_hf_cache
cache_paths = set_hf_cache("./cache")
from datasets import load_dataset

'''
Download Dataset form Hugging Face

Path:
BenchMark/Dataset/logs/<all_downloaded_data.txt>
'''

# Benchmark 
benchmarks = {
    "knowledge_extract": [
        #{"source" : "HF", "name" : "google/boolq",      "split" : "validation", "sub" : None},
        #{"source" : "HF", "name" : "facebook/belebele", "split" : "test",       "sub" : "eng_Latn"}, 
        #{"source" : "HF", "name" : "facebook/belebele", "split" : "test",       "sub" : "zho_Hant"}, 
        #{"source" : "HF", "name" : "google/xquad",      "split" : "validation", "sub" : "xquad.en"},
        #{"source" : "HF", "name" : "google/xquad",      "split" : "validation", "sub" : "xquad.zh"},
        ],
    "chating": [
        #{"source" : "HF_file", "name" : "socialiqa-train-dev", "split" : "train", "sub" : "dev.jsonl"},
        #{"source" : "HF", "name" : "google-research-datasets/natural_questions", "split" : "validation", "sub" : "default"}
        ],
    #"common_knowledge": ["mmlu", "global_mmlu_lite"],
    #"logic_thinking": ["mbpp", "gsm8k"],
    #"translate": ["wmt14"],
    "RAG": [
        {"source" : "HF", "name" : "rag-datasets/rag-mini-wikipedia", "split" : "passages", "sub" : "text-corpus"}, # RAG QA
        {"source" : "HF", "name" : "rag-datasets/rag-mini-wikipedia", "split" : "test",    "sub" : "question-answer"}, # RAG QA
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
                cache_dir=cache_paths["DATASET_CACHE"],
            )
            msg += f"Samples [{len(ds)}]"
        elif source == "HF_file":
            file_path = os.path.join(cache_paths["DATASET_CACHE"], name, sub)
            ds = load_dataset(
                "json",
                data_files=file_path,
                split=split,
                cache_dir=cache_paths["DATASET_CACHE"],
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
    preview_amount = 10
    for category, dataset_info in benchmarks.items():
        for dataset in dataset_info:
            source = dataset['source']
            name   = dataset['name']
            sub    = dataset['sub']
            split    = dataset['split']
            ds = preview_dataset(source, name, split, sub)
            log_name = remove_special_symbol(f"{name}-{sub}")
            log = LogWriter(log_name=f"{log_name}", log_folder_name="logs", root_folder="Dataset")
            log.clear()
            if ds is not None:
                for idx in range(min(len(ds), preview_amount)):
                    print(ds[idx])
                    log.write_log(ds[idx])
                

if __name__ == "__main__":
    main()
