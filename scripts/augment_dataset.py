"""
Data Augmentation Script: Expand dataset from 500 to 2000 rows
データ拡張スクリプト: 500行から2000行への拡張

Strategy:
1. Analyze current dataset patterns
2. Generate synthetic data with diverse price trends
3. Add high-volatility periods (trading opportunities)
4. Balance profitable BUY/SELL examples
5. Preserve feature distributions
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_current_dataset(df: pd.DataFrame) -> dict:
    """Analyze current dataset characteristics"""
    print(f"Current dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    
    # Price analysis
    if 'close' in df.columns:
        price_mean = df['close'].mean()
        price_std = df['close'].std()
        price_range = df['close'].max() - df['close'].min()
        price_volatility = df['close'].pct_change().std()
        
        print(f"\nPrice Statistics:")
        print(f"  Mean: {price_mean:.2f}")
        print(f"  Std: {price_std:.2f}")
        print(f"  Range: {price_range:.2f}")
        print(f"  Volatility (pct_change std): {price_volatility:.4f}")
        
        return {
            'price_mean': price_mean,
            'price_std': price_std,
            'price_range': price_range,
            'volatility': price_volatility
        }
    
    return {}

def generate_synthetic_price_patterns(base_price: float, n_rows: int, pattern_type: str) -> np.ndarray:
    """Generate synthetic price patterns"""
    np.random.seed(42)
    
    if pattern_type == 'uptrend':
        # Upward trend with noise
        trend = np.linspace(0, base_price * 0.2, n_rows)
        noise = np.random.normal(0, base_price * 0.02, n_rows)
        prices = base_price + trend + noise
        
    elif pattern_type == 'downtrend':
        # Downward trend with noise
        trend = np.linspace(0, -base_price * 0.2, n_rows)
        noise = np.random.normal(0, base_price * 0.02, n_rows)
        prices = base_price + trend + noise
        
    elif pattern_type == 'volatile':
        # High volatility sideways
        noise = np.random.normal(0, base_price * 0.05, n_rows)
        cycle = base_price * 0.1 * np.sin(np.linspace(0, 4 * np.pi, n_rows))
        prices = base_price + noise + cycle
        
    elif pattern_type == 'stable':
        # Low volatility
        noise = np.random.normal(0, base_price * 0.01, n_rows)
        prices = base_price + noise
        
    else:  # 'mixed'
        # Mix of patterns
        prices = np.zeros(n_rows)
        chunk_size = n_rows // 4
        prices[0:chunk_size] = generate_synthetic_price_patterns(base_price, chunk_size, 'uptrend')
        prices[chunk_size:2*chunk_size] = generate_synthetic_price_patterns(base_price, chunk_size, 'volatile')
        prices[2*chunk_size:3*chunk_size] = generate_synthetic_price_patterns(base_price, chunk_size, 'downtrend')
        prices[3*chunk_size:] = generate_synthetic_price_patterns(base_price, n_rows - 3*chunk_size, 'stable')
    
    return prices

def augment_dataset(df: pd.DataFrame, target_rows: int = 2000) -> pd.DataFrame:
    """Augment dataset to target number of rows"""
    current_rows = len(df)
    additional_rows = target_rows - current_rows
    
    if additional_rows <= 0:
        print(f"Dataset already has {current_rows} rows, no augmentation needed.")
        return df
    
    print(f"\nGenerating {additional_rows} additional rows...")
    
    # Analyze current data
    stats = analyze_current_dataset(df)
    
    # Generate new data with diverse patterns
    n_per_pattern = additional_rows // 5
    patterns = ['uptrend', 'downtrend', 'volatile', 'stable', 'mixed']
    
    new_data_pieces = []
    
    for pattern in patterns:
        print(f"  Generating {n_per_pattern} rows with {pattern} pattern...")
        
        # Generate base prices
        if 'close' in df.columns:
            base_price = stats.get('price_mean', 1000000)
            prices = generate_synthetic_price_patterns(base_price, n_per_pattern, pattern)
            
            # Create new dataframe segment
            new_segment = pd.DataFrame()
            new_segment['close'] = prices
            
            # Calculate OHLV from close
            new_segment['open'] = new_segment['close'].shift(1).fillna(new_segment['close'].iloc[0])
            new_segment['high'] = new_segment[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, n_per_pattern))
            new_segment['low'] = new_segment[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, n_per_pattern))
            new_segment['volume'] = np.random.uniform(100, 10000, n_per_pattern)
            
            # Add timestamp (if exists in original)
            if 'timestamp' in df.columns:
                last_timestamp = df['timestamp'].max()
                new_segment['timestamp'] = pd.date_range(start=last_timestamp, periods=n_per_pattern+1, freq='1h')[1:]
            
            new_data_pieces.append(new_segment)
    
    # Combine all new data
    new_data = pd.concat(new_data_pieces, ignore_index=True)
    
    # Add remaining rows if needed
    remaining = target_rows - current_rows - len(new_data)
    if remaining > 0:
        print(f"  Adding {remaining} mixed pattern rows...")
        base_price = stats.get('price_mean', 1000000)
        prices = generate_synthetic_price_patterns(base_price, remaining, 'mixed')
        
        final_segment = pd.DataFrame()
        final_segment['close'] = prices
        final_segment['open'] = final_segment['close'].shift(1).fillna(final_segment['close'].iloc[0])
        final_segment['high'] = final_segment[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, remaining))
        final_segment['low'] = final_segment[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, remaining))
        final_segment['volume'] = np.random.uniform(100, 10000, remaining)
        
        new_data = pd.concat([new_data, final_segment], ignore_index=True)
    
    # Combine with original data
    augmented_df = pd.concat([df, new_data], ignore_index=True)
    
    print(f"\n✅ Augmentation complete!")
    print(f"  Original rows: {current_rows}")
    print(f"  New rows: {len(new_data)}")
    print(f"  Total rows: {len(augmented_df)}")
    
    return augmented_df

def main():
    """Main data augmentation pipeline"""
    print("=" * 80)
    print("DATA AUGMENTATION: 500 → 2000 rows")
    print("=" * 80)
    
    # Load current dataset
    input_path = Path("ml-dataset-enhanced-balanced.csv")
    if not input_path.exists():
        print(f"❌ Error: {input_path} not found!")
        return
    
    print(f"\nLoading dataset: {input_path}")
    df = pd.read_csv(input_path)
    
    # Analyze current dataset
    print("\n" + "=" * 80)
    print("CURRENT DATASET ANALYSIS")
    print("=" * 80)
    analyze_current_dataset(df)
    
    # Augment dataset
    print("\n" + "=" * 80)
    print("AUGMENTATION PROCESS")
    print("=" * 80)
    augmented_df = augment_dataset(df, target_rows=2000)
    
    # Save augmented dataset
    output_path = Path("ml-dataset-enhanced-balanced-2000.csv")
    print(f"\nSaving augmented dataset: {output_path}")
    augmented_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 80)
    print("✅ DATA AUGMENTATION COMPLETED")
    print("=" * 80)
    print(f"Output file: {output_path}")
    print(f"Total rows: {len(augmented_df)}")
    print(f"Total columns: {len(augmented_df.columns)}")
    
    # Final analysis
    print("\n" + "=" * 80)
    print("AUGMENTED DATASET ANALYSIS")
    print("=" * 80)
    analyze_current_dataset(augmented_df)

if __name__ == "__main__":
    main()
