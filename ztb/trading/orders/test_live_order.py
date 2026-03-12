#!/usr/bin/env python3
"""
Real trading test script for Coincheck - Place a test buy order with very low price.
This script demonstrates actual API calls to Coincheck for placing orders.
"""

import hashlib
import hmac
import json
import os
import sys
import time
import urllib.parse
from pathlib import Path

import requests
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def get_current_price() -> float:
    """Get current BTC/JPY price from Coincheck public API."""
    response = requests.get("https://coincheck.com/api/ticker", timeout=10)
    response.raise_for_status()
    data = response.json()
    return float(data["last"])

def place_buy_order(api_key: str, api_secret: str, price: float, amount: float) -> dict:
    """Place a buy order on Coincheck."""
    url = "https://coincheck.com/api/exchange/orders"

    # Prepare order data - Coincheck expects URL-encoded form data
    order_data = {
        "pair": "btc_jpy",
        "order_type": "buy",
        "rate": str(int(price)),  # Price as string
        "amount": str(amount),  # Amount as string
    }
    body = urllib.parse.urlencode(order_data)

    # Create signature using the exact payload string that will be sent
    nonce = str(int(time.time() * 1000000))
    message = nonce + url + body
    signature = create_signature(api_secret, message)

    headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-NONCE": nonce,
        "ACCESS-SIGNATURE": signature,
        "Content-Type": "application/x-www-form-urlencoded",
    }

    response = requests.post(url, headers=headers, data=body, timeout=10)

    if response.status_code == 200:
        return response.json()
    else:
        error_msg = f"API Error {response.status_code}: {response.text}"
        raise Exception(error_msg)

def main():
    """Main function to place a test buy order."""
    try:
        # Load environment variables from .env file
        env_path = PROJECT_ROOT / ".env"
        load_dotenv(env_path)

        print("🚀 Starting real Coincheck trading test")

        # Get API credentials
        api_key = os.getenv("COINCHECK_API_KEY", "").strip()
        api_secret = os.getenv("COINCHECK_API_SECRET", "").strip()

        if not api_key or not api_secret:
            print("❌ No API credentials found!")
            print("   Please set environment variables:")
            print("   COINCHECK_API_KEY=your_api_key")
            print("   COINCHECK_API_SECRET=your_api_secret")
            return 1

        print("✅ API credentials found")

        # Get current price
        print("📊 Getting current BTC/JPY price...")
        current_price = get_current_price()
        print(f"💰 Current BTC/JPY price: ¥{current_price:,.0f}")

        # Calculate very low price for testing (should be rejected)
        test_price = current_price * 0.1  # 10% of current price
        test_amount = 0.001  # Very small amount (0.001 BTC)

        print("\n🎯 Test Order Details:")
        print("   Side: BUY")
        print(f"   Amount: {test_amount} BTC")
        print(f"   Price: ¥{test_price:,.0f}")
        print("   Expected: REJECTED (insufficient funds or invalid price)")

        # Confirm before placing order
        print("\n⚠️  WARNING: This will place a REAL order on Coincheck!")
        confirm = (
            input("   Do you want to continue? (type 'yes' to confirm): ")
            .strip()
            .lower()
        )

        if confirm != "yes":
            print("❌ Order cancelled by user")
            return 0

        print("📡 Placing buy order...")

        # Place the order
        result = place_buy_order(api_key, api_secret, test_price, test_amount)

        print("✅ Order placed successfully!")
        print(f"� Order Response: {json.dumps(result, indent=2)}")

        if "id" in result:
            print(f"🆔 Order ID: {result['id']}")
        if "success" in result and result["success"]:
            print("🎉 Order appears to be accepted")
        else:
            print("⚠️  Order may have been rejected")

        print("\n🔍 Check your Coincheck account to verify the order status")
        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
