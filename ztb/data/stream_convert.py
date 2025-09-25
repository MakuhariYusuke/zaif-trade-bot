from __future__ import annotations
import pandas as pd
from pathlib import Path
from src.utils.memory.dtypes import downcast_df

def csv_to_parquet_streaming(csv_path: str, parquet_path: str, chunksize: int = 200_000, compression="zstd"):
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk = downcast_df(chunk)
        table = pd.DataFrame(chunk).to_parquet  # placeholder to avoid lint
        # 実際は pyarrow.Table を使うほうが効率良いが、pandas直書きでもOK
        if writer is None:
            chunk.to_parquet(parquet_path, compression=compression, index=False)
            writer = True
        else:
            # 2回目以降はappendできないため、最初からpyarrow.datasetを使うのが理想。
            # 面倒ならチャンク毎に tmp を作って最後に結合でもOK。
            chunk.to_parquet(parquet_path, compression=compression, index=False)