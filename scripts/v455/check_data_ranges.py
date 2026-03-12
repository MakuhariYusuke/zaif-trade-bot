
import pandas as pd
from pathlib import Path

data_dir = Path("c:/Users/Admin/dev/zaif-trade-bot/data")
files = [
    "btc_jpy_1m_v454.csv",
    "btc_jpy_1m_latest.csv",
    "btc_jpy_1m_yfinance.csv",
    "btc_jpy_1m_merged.csv"
]

for f in files:
    p = data_dir / f
    if p.exists():
        try:
            print(f"Checking {f}...")
            # Read only index to be fast
            df = pd.read_csv(p, index_col=0)
            # Convert index to datetime
            df.index = pd.to_datetime(df.index, utc=True)
            
            print(f"File: {f}")
            print(f"  Start: {df.index.min()}")
            print(f"  End:   {df.index.max()}")
            print(f"  Rows:  {len(df)}")
            print("-" * 20)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    else:
        print(f"File not found: {f}")
