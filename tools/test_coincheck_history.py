import pandas as pd
import requests

url = "https://coincheck.com/api/charts/candle_rates"
# Try to fetch a large number of candles
params = {"pair": "btc_jpy", "unit": 60, "limit": 1000}  # 1 minute  # Try 1000 first

print(f"Fetching with limit={params['limit']}...")
response = requests.get(url, params=params)
if response.status_code == 200:
    data = response.json()
    print(f"Received {len(data)} candles")
    if len(data) > 0:
        first_candle = data[0]
        last_candle = data[-1]
        print(f"First candle: {pd.to_datetime(first_candle[0], unit='s')}")
        print(f"Last candle: {pd.to_datetime(last_candle[0], unit='s')}")
else:
    print(f"Error: {response.status_code}")

# Try with a much larger limit
params["limit"] = 10000
print(f"\nFetching with limit={params['limit']}...")
response = requests.get(url, params=params)
if response.status_code == 200:
    data = response.json()
    print(f"Received {len(data)} candles")
    if len(data) > 0:
        first_candle = data[0]
        last_candle = data[-1]
        print(f"First candle: {pd.to_datetime(first_candle[0], unit='s')}")
        print(f"Last candle: {pd.to_datetime(last_candle[0], unit='s')}")
else:
    print(f"Error: {response.status_code}")
