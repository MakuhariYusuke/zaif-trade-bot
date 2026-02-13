
import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def download_btc_jpy_1m():
    print("Downloading latest BTC-JPY 1m data from Yahoo Finance...")
    
    # Yahoo Finance allows max 7 days for 1m interval
    # We'll fetch the max available
    ticker = "BTC-JPY"
    try:
        data = yf.download(ticker, period="7d", interval="1m", progress=False)
    except Exception as e:
        print(f"Error downloading data: {e}")
        return
    
    if data.empty:
        print("Error: No data downloaded.")
        return

    # Reset index to make timestamp a column
    data = data.reset_index()
    
    # Rename columns to match system format (lowercase)
    # yfinance returns: Date/Datetime, Open, High, Low, Close, Adj Close, Volume
    # We need: timestamp, open, high, low, close, volume
    
    # Check column names (yfinance format can vary)
    # print(f"Columns found: {data.columns.tolist()}")
    
    # Normalize columns
    # Handle MultiIndex columns if present (yfinance update)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [c.lower() for c in data.columns]
    
    # Rename 'datetime' or 'date' to 'timestamp'
    if 'datetime' in data.columns:
        data = data.rename(columns={'datetime': 'timestamp'})
    elif 'date' in data.columns:
        data = data.rename(columns={'date': 'timestamp'})
        
    # Ensure required columns exist
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    available = [c for c in required if c in data.columns]
    
    if len(available) < len(required):
        print(f"Warning: Missing columns. Found: {available}")
            
    # Select and reorder
    final_df = data[available].copy()
    
    # Save
    output_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"btc_jpy_1m_latest_7d_{timestamp_str}.csv")
    # Also save as a fixed name for easy access
    fixed_filename = os.path.join(output_dir, "btc_jpy_1m_latest.csv")
    
    final_df.to_csv(filename, index=False)
    final_df.to_csv(fixed_filename, index=False)
    
    print(f"Successfully saved {len(final_df)} rows to:")
    print(f"- {filename}")
    print(f"- {fixed_filename}")
    
    # Basic stats
    print("\nData Summary:")
    print(f"Start: {final_df['timestamp'].min()}")
    print(f"End:   {final_df['timestamp'].max()}")
    print(f"Rows:  {len(final_df)}")

if __name__ == "__main__":
    download_btc_jpy_1m()
