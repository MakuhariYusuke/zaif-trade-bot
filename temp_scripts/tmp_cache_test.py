import tempfile
from pathlib import Path

import pandas as pd

from ztb.cache.data_loader import DataLoader

p = tempfile.TemporaryDirectory()
cache_dir = Path(p.name) / "cache"
cache_dir.mkdir()

loader = DataLoader(cache_dir=str(cache_dir))

test_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

result = loader.load_with_cache("test_key", lambda: test_df)
print("result_equals", result.equals(test_df))
cache_file = cache_dir / "test_key.pkl"
print("cache_exists", cache_file.exists())
if cache_file.exists():
    print("cache_size", cache_file.stat().st_size)
else:
    print("no file")
