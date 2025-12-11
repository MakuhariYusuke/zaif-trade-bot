import requests


def test_coincheck_candles():
    url = "https://coincheck.com/api/charts/candle_rates"
    params = {"pair": "btc_jpy", "unit": 60, "limit": 1440}
    try:
        response = requests.get(url, params=params)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Data type: {type(data)}")
            print(f"Data length: {len(data)}")
            if len(data) > 0:
                print("Sample:", data[0])
                print("Sample -1:", data[-1])
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_coincheck_candles()
