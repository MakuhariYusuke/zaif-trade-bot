import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PotentialAlpha")

def calculate_potential_profit(data_path):
    df = pd.read_csv(data_path, parse_dates=True, index_col=0)
    logger.info(f"Loaded {len(df)} rows from {data_path}")
    
    # Simple ZigZag-like approach to find peaks and valleys
    # We want to know: specific "Ideal" profit if we bought every localized bottom and sold every localized top
    # Threshold for a "swing" (e.g., 0.5% move)
    
    close = df['close'].values
    timestamps = df.index
    
    threshold_pct = 0.005 # 0.5% swing required to mark a pivot
    
    pivots = [] # (index, price, type) type=1 for peak, -1 for valley
    
    # Initialize
    trend = 0 
    last_pivot_price = close[0]
    last_pivot_idx = 0
    
    pivots.append((0, close[0], 0)) # Start point
    
    for i in range(1, len(close)):
        price = close[i]
        diff = (price - last_pivot_price) / last_pivot_price
        
        if trend == 0:
            if diff > threshold_pct:
                trend = 1
                last_pivot_idx = i
                last_pivot_price = price
            elif diff < -threshold_pct:
                trend = -1
                last_pivot_idx = i
                last_pivot_price = price
        elif trend == 1: # Uptrend, looking for peak
            if price > last_pivot_price:
                last_pivot_price = price
                last_pivot_idx = i
            elif diff < -threshold_pct: # Reversed
                pivots.append((last_pivot_idx, last_pivot_price, 1)) # Record Peak
                trend = -1
                last_pivot_price = price
                last_pivot_idx = i
        elif trend == -1: # Downtrend, looking for valley
            if price < last_pivot_price:
                last_pivot_price = price
                last_pivot_idx = i
            elif diff > threshold_pct: # Reversed
                pivots.append((last_pivot_idx, last_pivot_price, -1)) # Record Valley
                trend = 1
                last_pivot_price = price
                last_pivot_idx = i
                
    # Calculate Theoretical Stats
    total_potential_pnl = 0
    trades = 0
    
    # Assuming we buy at Valley and Sell at Peak
    # Filter pivots to alternating Valley -> Peak
    
    clean_trades = []
    
    last_type = 0 # 0=none, -1=valley/buy, 1=peak/sell
    entry_price = 0
    
    for idx, price, p_type in pivots:
        if p_type == -1: # Valley (Buy signal)
            if last_type != -1: # Valid entry
                entry_price = price
                last_type = -1
        elif p_type == 1: # Peak (Sell signal)
            if last_type == -1: # Valid exit
                pnl = price - entry_price
                pnl_pct = pnl / entry_price
                total_potential_pnl += pnl
                trades += 1
                clean_trades.append({
                    "entry_price": entry_price,
                    "exit_price": price,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct
                })
                last_type = 1
                
    # Report
    initial_price = close[0]
    final_price = close[-1]
    buy_hold_pnl = final_price - initial_price
    
    logger.info(f"Total Theoretical Trades (Swing > {threshold_pct*100}%): {trades}")
    logger.info(f"Total Potential Price Diff Sum (per unit): ¥{total_potential_pnl:,.0f}")
    
    # Assume 1BTC trade for simplicity in Price Diff Sum
    # Actually user trades 0.01 BTC or similar.
    
    logger.info(f"Buy & Hold PnL (per unit): ¥{buy_hold_pnl:,.0f}")
    if buy_hold_pnl != 0:
        logger.info(f"Gain Multiplier vs BuyHold: {total_potential_pnl / abs(buy_hold_pnl):.2f}x")

    logger.info("--- Top 5 Trades ---")
    sorted_trades = sorted(clean_trades, key=lambda x: x['pnl'], reverse=True)
    for t in sorted_trades[:5]:
        logger.info(f"Gain: ¥{t['pnl']:,.0f} ({t['pnl_pct']*100:.2f}%)")

if __name__ == "__main__":
    calculate_potential_profit("data/yahoo_finance/btc_jpy_1m.csv")
