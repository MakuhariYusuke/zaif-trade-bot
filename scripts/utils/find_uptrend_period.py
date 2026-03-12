rimport pandas as pd

def find_best_uptrend(file_path, window_hours=24):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    best_return = -float('inf')
    best_start = None
    best_end = None

    # Resample to 1 minute to ensure regular intervals if needed, or just iterate
    # Since it's 1m data, we can just look at windows of N rows (60 * window_hours)

    window_size = 60 * window_hours

    if len(df) < window_size:
        print("Not enough data for the requested window.")
        return

    # Optimization: Check every 60 minutes instead of every minute to speed up
    step = 60

    for i in range(0, len(df) - window_size, step):
        start_price = df.iloc[i]['close']
        end_price = df.iloc[i + window_size]['close']

        if start_price == 0: continue

        pct_change = (end_price - start_price) / start_price

        if pct_change > best_return:
            best_return = pct_change
            best_start = df.index[i]
            best_end = df.index[i + window_size]

    print(f"Best {window_hours}h Uptrend:")
    print(f"Start: {best_start}")
    print(f"End: {best_end}")
    print(f"Return: {best_return * 100:.2f}%")

if __name__ == "__main__":
    find_best_uptrend('data/yahoo_finance/btc_jpy_1m_converted.csv', window_hours=24)
