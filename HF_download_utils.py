import os
from typing import Dict, Literal

def set_hf_cache(path: str) -> Dict[Literal["PATH_ROOT", "BASE_CACHE", "DATASET_CACHE", "HUB_CACHE", "TRANSFORMER_CACHE"], str]:
    ''' Set Hugging face download cache file to path, it'll create sub folder: [huggingface, huggingface/datasets, huggingface/hub, huggingface/transformers] to download files '''
    assert os.path.isdir(path), f"Invalid folder, please create one: {path}"
    PATH_ROOT     = os.path.abspath(path)
    BASE_CACHE    = os.path.join(PATH_ROOT, "huggingface")
    DATASET_CACHE = os.path.join(BASE_CACHE, "datasets")
    HUB_CACHE     = os.path.join(BASE_CACHE, "hub")
    TRANSFORMER_CACHE = os.path.join(BASE_CACHE, "transformers")
    os.makedirs(DATASET_CACHE, exist_ok=True)
    os.makedirs(HUB_CACHE, exist_ok=True)
    os.makedirs(TRANSFORMER_CACHE, exist_ok=True)
    os.environ["HF_HOME"] = BASE_CACHE
    os.environ["HF_DATASETS_CACHE"] = DATASET_CACHE
    os.environ["HUGGINGFACE_HUB_CACHE"] = HUB_CACHE
    os.environ["HF_HUB_CACHE"] = HUB_CACHE
    os.environ["TRANSFORMERS_CACHE"] = TRANSFORMER_CACHE
    os.environ["XDG_CACHE_HOME"] = BASE_CACHE
    return {
        "PATH_ROOT"         : PATH_ROOT,
        "BASE_CACHE"        : BASE_CACHE,
        "DATASET_CACHE"     : DATASET_CACHE,
        "HUB_CACHE"         : HUB_CACHE,
        "TRANSFORMER_CACHE" : TRANSFORMER_CACHE,
    }

if __name__ == "__main__":
    print(set_hf_cache("./temp"))