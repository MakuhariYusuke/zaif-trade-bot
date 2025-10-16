#!/usr/bin/env python3
"""
Real trading test script for Coincheck and bitFlyer - Complete trading test with order placement and cancellation.
This script demonstrates actual API calls to selected exchange for placing and canceling orders.
"""

import sys
import os
import hmac
import hashlib
import time
import requests
import json
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from ztb.utils.errors import InsufficientFundsError, MinimumSizeError

# Load environment variables from .env file
load_dotenv()

def create_signature(secret: str, message: str) -> str:
    """Create HMAC-SHA256 signature for Coincheck API."""
    return hmac.new(
        secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def create_bitflyer_signature(secret: str, timestamp: str, method: str, path: str, body: str = "") -> str:
    """Create HMAC-SHA256 signature for bitFlyer API."""
    message = timestamp + method + path + body
    return hmac.new(
        secret.encode('utf-8'),
        message.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()

def get_current_price(exchange: str) -> float:
    """Get current BTC/JPY price from selected exchange public API."""
    if exchange.lower() == "coincheck":
        response = requests.get("https://coincheck.com/api/ticker", timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data["last"])
    elif exchange.lower() == "bitflyer":
        response = requests.get("https://api.bitflyer.com/v1/ticker?product_code=BTC_JPY", timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data["ltp"])
    else:
        raise ValueError(f"Unsupported exchange: {exchange}")

def place_buy_order(api_key: str, api_secret: str, price: float, amount: float) -> dict:
    """Place a buy order on Coincheck."""
    url = "https://coincheck.com/api/exchange/orders"

    # Prepare order data
    order_data = {
        "pair": "btc_jpy",
        "order_type": "buy",
        "rate": str(int(price)),  # Price as string
        "amount": str(amount),    # Amount as string
        "market_buy_amount": None,
        "position_id": None,
        "stop_loss_rate": None
    }

    # Create signature
    nonce = str(int(time.time() * 1000000))
    message = nonce + url + json.dumps(order_data, separators=(',', ':'))
    signature = create_signature(api_secret, message)

    headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-NONCE": nonce,
        "ACCESS-SIGNATURE": signature,
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=order_data, timeout=10)

    if response.status_code == 200:
        return response.json()
    else:
        error_msg = f"API Error {response.status_code}: {response.text}"
        # Check for specific error types
        if "insufficient" in error_msg.lower() or "balance" in error_msg.lower():
            raise InsufficientFundsError(f"Insufficient funds for buy order: {error_msg}")
        elif "minimum" in error_msg.lower() or "size" in error_msg.lower():
            raise MinimumSizeError(f"Order size below minimum: {error_msg}")
        else:
            raise Exception(error_msg)

def place_sell_order(api_key: str, api_secret: str, price: float, amount: float) -> dict:
    """Place a sell order on Coincheck."""
    url = "https://coincheck.com/api/exchange/orders"

    # Prepare order data
    order_data = {
        "pair": "btc_jpy",
        "order_type": "sell",
        "rate": str(int(price)),  # Price as string
        "amount": str(amount),    # Amount as string
        "market_buy_amount": None,
        "position_id": None,
        "stop_loss_rate": None
    }

    # Create signature
    nonce = str(int(time.time() * 1000000))
    message = nonce + url + json.dumps(order_data, separators=(',', ':'))
    signature = create_signature(api_secret, message)

    headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-NONCE": nonce,
        "ACCESS-SIGNATURE": signature,
        "Content-Type": "application/json"
    }

    response = requests.post(url, headers=headers, json=order_data, timeout=10)

    if response.status_code == 200:
        return response.json()
    else:
        error_msg = f"API Error {response.status_code}: {response.text}"
        raise Exception(error_msg)

def cancel_order(api_key: str, api_secret: str, order_id: str) -> dict:
    """Cancel an order on Coincheck."""
    url = f"https://coincheck.com/api/exchange/orders/{order_id}"

    # Create signature
    nonce = str(int(time.time() * 1000000))
    message = nonce + url
    signature = create_signature(api_secret, message)

    headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-NONCE": nonce,
        "ACCESS-SIGNATURE": signature,
        "Content-Type": "application/json"
    }

    response = requests.delete(url, headers=headers, timeout=10)

    if response.status_code == 200:
        return response.json()
    else:
        error_msg = f"API Error {response.status_code}: {response.text}"
        raise Exception(error_msg)

def get_balance(api_key: str, api_secret: str) -> dict:
    """Get account balance from Coincheck."""
    url = "https://coincheck.com/api/accounts/balance"

    # Create signature
    nonce = str(int(time.time() * 1000000))
    message = nonce + url
    signature = create_signature(api_secret, message)

    headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-NONCE": nonce,
        "ACCESS-SIGNATURE": signature,
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 200:
        return response.json()
    else:
        error_msg = f"API Error {response.status_code}: {response.text}"
        raise Exception(error_msg)

# bitFlyer API functions
def place_buy_order_bitflyer(api_key: str, api_secret: str, price: float, amount: float) -> dict:
    """Place a buy order on bitFlyer."""
    url = "https://api.bitflyer.com"
    path = "/v1/me/sendchildorder"
    method = "POST"

    # Prepare order data
    order_data = {
        "product_code": "BTC_JPY",
        "child_order_type": "LIMIT",
        "side": "BUY",
        "price": int(price),
        "size": amount
    }
    body = json.dumps(order_data, separators=(',', ':'))

    # Create signature
    timestamp = str(int(time.time() * 1000))
    signature = create_bitflyer_signature(api_secret, timestamp, method, path, body)

    headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-SIGN": signature,
        "Content-Type": "application/json"
    }

    response = requests.post(url + path, headers=headers, data=body, timeout=10)

    if response.status_code == 200:
        return response.json()
    else:
        error_msg = f"API Error {response.status_code}: {response.text}"
        raise Exception(error_msg)

def place_sell_order_bitflyer(api_key: str, api_secret: str, price: float, amount: float) -> dict:
    """Place a sell order on bitFlyer."""
    url = "https://api.bitflyer.com"
    path = "/v1/me/sendchildorder"
    method = "POST"

    # Prepare order data
    order_data = {
        "product_code": "BTC_JPY",
        "child_order_type": "LIMIT",
        "side": "SELL",
        "price": int(price),
        "size": amount
    }
    body = json.dumps(order_data, separators=(',', ':'))

    # Create signature
    timestamp = str(int(time.time() * 1000))
    signature = create_bitflyer_signature(api_secret, timestamp, method, path, body)

    headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-SIGN": signature,
        "Content-Type": "application/json"
    }

    response = requests.post(url + path, headers=headers, data=body, timeout=10)

    if response.status_code == 200:
        return response.json()
    else:
        error_msg = f"API Error {response.status_code}: {response.text}"
        raise Exception(error_msg)

def cancel_order_bitflyer(api_key: str, api_secret: str, order_id: str) -> dict:
    """Cancel an order on bitFlyer."""
    url = "https://api.bitflyer.com"
    path = "/v1/me/cancelchildorder"
    method = "POST"

    # Prepare cancel data
    cancel_data = {
        "child_order_acceptance_id": order_id
    }
    body = json.dumps(cancel_data, separators=(',', ':'))

    # Create signature
    timestamp = str(int(time.time() * 1000))
    signature = create_bitflyer_signature(api_secret, timestamp, method, path, body)

    headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-SIGN": signature,
        "Content-Type": "application/json"
    }

    response = requests.post(url + path, headers=headers, data=body, timeout=10)

    if response.status_code == 200:
        return {"success": True, "message": "Order cancelled"}
    else:
        error_msg = f"API Error {response.status_code}: {response.text}"
        raise Exception(error_msg)

def get_balance_bitflyer(api_key: str, api_secret: str) -> dict:
    """Get account balance from bitFlyer."""
    url = "https://api.bitflyer.com"
    path = "/v1/me/getbalance"
    method = "GET"

    # Create signature
    timestamp = str(int(time.time() * 1000))
    signature = create_bitflyer_signature(api_secret, timestamp, method, path)

    headers = {
        "ACCESS-KEY": api_key,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-SIGN": signature,
        "Content-Type": "application/json"
    }

    response = requests.get(url + path, headers=headers, timeout=10)

    if response.status_code == 200:
        return response.json()
    else:
        error_msg = f"API Error {response.status_code}: {response.text}"
        raise Exception(error_msg)

def main():
    """Main function to perform complete trading test."""
    import argparse
    parser = argparse.ArgumentParser(description='Exchange trading test')
    parser.add_argument('--exchange', choices=['coincheck', 'bitflyer'], default='coincheck', help='Exchange to use (default: coincheck)')
    parser.add_argument('--dry-run', action='store_true', help='Run in dry-run mode (no real orders)')
    args = parser.parse_args()

    try:
        print(f"🚀 Starting complete {args.exchange.upper()} trading test")
        if args.dry_run:
            print("🔍 DRY-RUN MODE - No real orders will be placed")

        # Get API credentials based on exchange
        if args.exchange == 'coincheck':
            api_key = os.getenv("COINCHECK_API_KEY", "").strip()
            api_secret = os.getenv("COINCHECK_API_SECRET", "").strip()
            env_vars = ["COINCHECK_API_KEY", "COINCHECK_API_SECRET"]
        else:  # bitflyer
            api_key = os.getenv("BITFLYER_API_KEY", "").strip()
            api_secret = os.getenv("BITFLYER_API_SECRET", "").strip()
            env_vars = ["BITFLYER_API_KEY", "BITFLYER_API_SECRET"]

        if not api_key or not api_secret:
            if not args.dry_run:
                print("❌ No API credentials found!")
                print("   Please set environment variables:")
                for var in env_vars:
                    print(f"   {var}=your_{var.lower()}")
                print("   Or use --dry-run for testing")
                return 1
            else:
                print("🔍 Dry-run mode: using dummy credentials")
                api_key = "dummy_key"
                api_secret = "dummy_secret"
        else:
            print("✅ API credentials found")

        # Get current price
        print("📊 Getting current BTC/JPY price...")
        current_price = get_current_price(args.exchange)
        print(f"💰 Current BTC/JPY price: ¥{current_price:,.0f}")

        # Check initial balance
        print("\n💵 Checking initial account balance...")
        if args.dry_run:
            initial_balance = {
                "jpy": {"free": 100000.0, "locked": 0.0},
                "btc": {"free": 0.1, "locked": 0.0}
            }
            print("✅ Initial balance (simulated):")
            print(f"   JPY: ¥{initial_balance['jpy']['free']:,.0f} free, ¥{initial_balance['jpy']['locked']:,.0f} locked")
            print(f"   BTC: {initial_balance['btc']['free']:.6f} free, {initial_balance['btc']['locked']:.6f} locked")
        else:
            if args.exchange == 'coincheck':
                initial_balance = get_balance(api_key, api_secret)
                print("✅ Initial balance:")
                # Coincheck API returns balances as strings, not nested dicts
                jpy_balance = float(initial_balance.get("jpy", "0"))
                btc_balance = float(initial_balance.get("btc", "0"))
                print(f"   JPY: ¥{jpy_balance:,.0f}")
                print(f"   BTC: {btc_balance:.6f}")
            else:  # bitflyer
                initial_balance = get_balance_bitflyer(api_key, api_secret)
                print("✅ Initial balance:")
                # bitFlyer API returns list of balance objects
                jpy_balance = 0.0
                btc_balance = 0.0
                for balance_item in initial_balance:
                    currency = balance_item.get("currency_code", "").lower()
                    available = float(balance_item.get("available", 0))
                    if currency == "jpy":
                        jpy_balance = available
                    elif currency == "btc":
                        btc_balance = available
                print(f"   JPY: ¥{jpy_balance:,.0f}")
                print(f"   BTC: {btc_balance:.6f}")

        # Phase 1: Place a small buy order (should succeed if you have JPY balance)
        test_buy_price = current_price * 0.995  # 0.5% below current price
        
        # Set minimum order size based on exchange
        # Both exchanges use 0.001 BTC minimum for consistency
        test_amount = 0.001  # Minimum order size: 0.001 BTC (1 mBTC)

        print(f"\n🛒 Phase 1: Buy Order Test")
        print(f"   Amount: {test_amount} BTC")
        print(f"   Price: ¥{test_buy_price:,.0f}")
        print(f"   Total: ¥{test_buy_price * test_amount:,.0f}")
        print(f"   ⚠️  Minimum order size: {test_amount} BTC (1 mBTC)")

        # Confirm before placing order (skip in dry-run)
        if not args.dry_run:
            print(f"\n⚠️  ⚠️  ⚠️  CRITICAL WARNING ⚠️  ⚠️  ⚠️")
            print(f"   This will place REAL orders on {args.exchange.upper()}!")
            print("   - Buy order: very small amount for testing")
            print("   - Sell order: high price (will not execute)")
            print("   - Sell order will be cancelled immediately")
            print("   Make sure you have sufficient JPY balance!")
            print()
            confirm = input("   Do you want to continue? (type 'yes' to confirm): ").strip().lower()

            if confirm != 'yes':
                print("❌ Test cancelled by user")
                return 0
        else:
            print("🔍 Dry-run: skipping confirmation")

        print("📡 Placing buy order...")
        if args.dry_run:
            # Simulate successful order placement
            buy_result = {
                "id": "12345",
                "success": True,
                "message": "Simulated buy order placed"
            }
            print("✅ Buy order simulated successfully!")
            buy_order_id = "12345"
        else:
            try:
                if args.exchange == 'coincheck':
                    buy_result = place_buy_order(api_key, api_secret, test_buy_price, test_amount)
                else:  # bitflyer
                    buy_result = place_buy_order_bitflyer(api_key, api_secret, test_buy_price, test_amount)
                print("✅ Buy order placed successfully!")
                
                # Extract order ID
                buy_order_id = None
                if args.exchange == 'coincheck':
                    if 'id' in buy_result:
                        buy_order_id = str(buy_result['id'])
                else:  # bitflyer
                    if 'child_order_acceptance_id' in buy_result:
                        buy_order_id = str(buy_result['child_order_acceptance_id'])
                        
            except (InsufficientFundsError, MinimumSizeError) as e:
                error_msg = str(e)
                if isinstance(e, InsufficientFundsError):
                    print("⚠️  Buy order failed due to insufficient funds (expected for testing)")
                elif isinstance(e, MinimumSizeError):
                    print("⚠️  Buy order failed due to minimum size requirements")
                print("🔄 Continuing with test simulation...")
                buy_result = {
                    "error": type(e).__name__.lower(),
                    "message": str(e)
                }
                buy_order_id = None
            except Exception as e:
                # Re-raise unexpected errors
                raise

        print(f"🆔 Buy Order Response: {json.dumps(buy_result, indent=2)}")

        if buy_order_id:
            print(f"🆔 Buy Order ID: {buy_order_id}")

        # Wait a moment
        print("⏳ Waiting 2 seconds...")
        time.sleep(2)

        # Phase 2: Place a high-price sell order (should succeed but not execute)
        high_sell_price = current_price * 2.0  # 2x current price (very high)
        sell_amount = test_amount  # Same amount

        print(f"\n💰 Phase 2: High-Price Sell Order Test")
        print(f"   Amount: {sell_amount} BTC")
        print(f"   Price: ¥{high_sell_price:,.0f}")
        print(f"   Total: ¥{high_sell_price * sell_amount:,.0f}")
        print("   (This order should not execute due to high price)")

        print("📡 Placing high-price sell order...")
        if args.dry_run:
            # Simulate successful order placement
            sell_result = {
                "id": "12346",
                "success": True,
                "message": "Simulated sell order placed"
            }
            print("✅ Sell order simulated successfully!")
            sell_order_id = "12346"
        else:
            try:
                if args.exchange == 'coincheck':
                    sell_result = place_sell_order(api_key, api_secret, high_sell_price, sell_amount)
                else:  # bitflyer
                    sell_result = place_sell_order_bitflyer(api_key, api_secret, high_sell_price, sell_amount)
                print("✅ Sell order placed successfully!")
                
                # Extract order ID
                sell_order_id = None
                if args.exchange == 'coincheck':
                    if 'id' in sell_result:
                        sell_order_id = str(sell_result['id'])
                else:  # bitflyer
                    if 'child_order_acceptance_id' in sell_result:
                        sell_order_id = str(sell_result['child_order_acceptance_id'])
                        
            except (InsufficientFundsError, MinimumSizeError) as e:
                error_msg = str(e)
                if isinstance(e, InsufficientFundsError):
                    print("⚠️  Sell order failed due to insufficient funds (expected for testing)")
                elif isinstance(e, MinimumSizeError):
                    print("⚠️  Sell order failed due to minimum size requirements")
                print("🔄 Continuing with test simulation...")
                sell_result = {
                    "error": type(e).__name__.lower(),
                    "message": str(e)
                }
                sell_order_id = None
            except Exception as e:
                # Re-raise unexpected errors
                raise

        print(f"🆔 Sell Order Response: {json.dumps(sell_result, indent=2)}")

        if sell_order_id:
            print(f"🆔 Sell Order ID: {sell_order_id}")

        # Phase 3: Immediately cancel the sell order
        if sell_order_id:
            print(f"\n❌ Phase 3: Cancel Sell Order")
            print(f"   Cancelling order ID: {sell_order_id}")

            print("📡 Cancelling sell order...")
            if args.dry_run:
                # Simulate successful cancellation
                cancel_result = {
                    "success": True,
                    "message": "Simulated order cancelled"
                }
                print("✅ Sell order simulated cancellation!")
            else:
                try:
                    if args.exchange == 'coincheck':
                        cancel_result = cancel_order(api_key, api_secret, sell_order_id)
                    else:  # bitflyer
                        cancel_result = cancel_order_bitflyer(api_key, api_secret, sell_order_id)
                    print("✅ Sell order cancelled successfully!")
                except Exception as e:
                    error_msg = str(e)
                    if "Order not found" in error_msg or "already cancelled" in error_msg.lower() or "not found" in error_msg.lower():
                        print("⚠️  Order cancellation failed (order may have already executed or been cancelled)")
                        print("🔄 Continuing with test...")
                        cancel_result = {
                            "error": "order_not_found",
                            "message": "Order may have already been executed or cancelled"
                        }
                    else:
                        # Re-raise other errors
                        raise e

            print(f"🆔 Cancel Response: {json.dumps(cancel_result, indent=2)}")

        print("\n🎉 Complete trading test finished!")

        # Check final balance
        print("\n💵 Checking final account balance...")
        if args.dry_run:
            final_balance = {
                "jpy": {"free": 100000.0, "locked": 0.0},
                "btc": {"free": 0.1, "locked": 0.0}
            }
            print("✅ Final balance (simulated):")
            print(f"   JPY: ¥{final_balance['jpy']['free']:,.0f} free, ¥{final_balance['jpy']['locked']:,.0f} locked")
            print(f"   BTC: {final_balance['btc']['free']:.6f} free, {final_balance['btc']['locked']:.6f} locked")
        else:
            if args.exchange == 'coincheck':
                final_balance = get_balance(api_key, api_secret)
                print("✅ Final balance:")
                # Coincheck API returns balances as strings, not nested dicts
                jpy_balance = float(final_balance.get("jpy", "0"))
                btc_balance = float(final_balance.get("btc", "0"))
                print(f"   JPY: ¥{jpy_balance:,.0f}")
                print(f"   BTC: {btc_balance:.6f}")
            else:  # bitflyer
                final_balance = get_balance_bitflyer(api_key, api_secret)
                print("✅ Final balance:")
                # bitFlyer API returns list of balance objects
                jpy_balance = 0.0
                btc_balance = 0.0
                for balance_item in final_balance:
                    currency = balance_item.get("currency_code", "").lower()
                    available = float(balance_item.get("available", 0))
                    if currency == "jpy":
                        jpy_balance = available
                    elif currency == "btc":
                        btc_balance = available
                print(f"   JPY: ¥{jpy_balance:,.0f}")
                print(f"   BTC: {btc_balance:.6f}")

        if args.dry_run:
            print("🔍 This was a DRY-RUN - no real orders were placed")
        else:
            print(f"🔍 Check your {args.exchange.upper()} account to verify:")
            print(f"   - Buy order status (ID: {buy_order_id})")
            print(f"   - Sell order should be cancelled (ID: {sell_order_id})")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())