import numpy as np
import pandas as pd

def downcast_df(df: pd.DataFrame, float_dtype: str = "float32", int_dtype: str = "int32") -> pd.DataFrame:
    out = df.copy()
    for c in out.select_dtypes(include=["float64"]).columns:
        out[c] = out[c].astype(float_dtype)  # type: ignore
    for c in out.select_dtypes(include=["int64", "int32", "int16"]).columns:
        out[c] = pd.to_numeric(out[c], downcast="integer").astype(int_dtype)  # type: ignore
    # bool, category 変換（カード小なら）
    for c in out.select_dtypes(include=["object"]).columns:
        nunique = out[c].nunique(dropna=True)
        if 0 < nunique <= max(256, len(out)//100):  # 小さいカテゴリのみ
            out[c] = out[c].astype("category")
    return out