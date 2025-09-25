"""Optimized parquet loading (skeleton)"""
from pathlib import Path
import pandas as pd
from typing import List, Optional

ESSENTIAL_COLUMNS = ['timestamp','open','high','low','close','volume','price','bid','ask','spread']

def load_parquet_essential(path: str | Path, essential: Optional[List[str]] = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    cols = essential or ESSENTIAL_COLUMNS
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema.names)
        to_load = [c for c in cols if c in available]
        if to_load:
            return pd.read_parquet(path, columns=to_load)
        return pd.read_parquet(path)
    except Exception:
        return pd.read_parquet(path)
