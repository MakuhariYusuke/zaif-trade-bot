
import pandas as pd
from pathlib import Path
import glob

data_dir = Path("c:/Users/Admin/dev/zaif-trade-bot/data")
files = list(data_dir.glob("*1m*.csv"))

print(f"Found {len(files)} files.")

for p in files:
    try:
        # Read only index to be fast
        # Some files might not have index at col 0 or might be different format
        # We'll try standard format first
        df = pd.read_csv(p, index_col=0, nrows=5) # Check header first
        
        # If index looks like timestamp
        try:
            pd.to_datetime(df.index[0])
            # Read full index
            df = pd.read_csv(p, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
            
            start = df.index.min()
            end = df.index.max()
            
            # We are looking for data around Dec 19-20 2025
            target_start = pd.Timestamp("2025-12-19", tz="UTC")
            target_end = pd.Timestamp("2025-12-21", tz="UTC")
            
            if start <= target_end and end >= target_start:
                print(f"MATCH CANDIDATE: {p.name}")
                print(f"  Start: {start}")
                print(f"  End:   {end}")
                print(f"  Rows:  {len(df)}")
                print("-" * 20)
        except Exception:
            pass
    except Exception as e:
        pass
