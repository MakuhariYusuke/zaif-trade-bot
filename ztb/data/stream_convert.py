from __future__ import annotations

from pathlib import Path

import pandas as pd

from ztb.utils.memory.dtypes import downcast_df


def csv_to_parquet_streaming(
    csv_path: str,
    parquet_path: str,
    chunksize: int = 200_000,
    compression: str = "zstd",
) -> None:
    """
    CSVファイルをチャンク単位で読み込み、Parquet形式で保存します。

    Args:
        csv_path (str): 入力CSVファイルのパス。
        parquet_path (str): 出力Parquetファイルのパス。
        chunksize (int, optional): 1チャンクあたりの行数。デフォルトは200,000。
        compression (str, optional): Parquetファイルの圧縮方式。デフォルトは"zstd"。

    Note:
        2回目以降のチャンクはappendできないため、全データを1つのParquetファイルにまとめる場合は
        pyarrow.dataset等の利用や、チャンク毎に一時ファイルを作成して後で結合する必要があります。
    """
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    parquet_path_obj = Path(parquet_path)
    parquet_path_obj.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk = downcast_df(chunk)
        table = pa.Table.from_pandas(chunk)
        if writer is None:
            writer = pq.ParquetWriter(
                str(parquet_path_obj), table.schema, compression=compression
            )
        writer.write_table(table)
    if writer is not None:
        writer.close()
