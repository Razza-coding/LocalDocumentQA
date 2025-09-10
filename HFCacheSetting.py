import os
from typing import Dict, Literal
import logging
class _IgnoreUA(logging.Filter):
    def filter(self, record):
        return "Using `TRANSFORMERS_CACHE` is deprecated" not in record.getMessage()
logging.getLogger().addFilter(_IgnoreUA())
import warnings
warnings.filterwarnings("ignore", message=r".*Using `TRANSFORMERS_CACHE` is deprecated*")

def set_hf_cache(path: str) -> Dict[Literal["PATH_ROOT", "BASE_CACHE", "DATASET_CACHE", "HUB_CACHE", "TRANSFORMER_CACHE"], str]:
    ''' Set Hugging face download cache file to path, it'll create sub folder: [huggingface, huggingface/datasets, huggingface/hub, huggingface/transformers] to download files '''
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
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
    print(set_hf_cache("./cache"))